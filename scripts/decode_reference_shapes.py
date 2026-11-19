#!/usr/bin/env python3
"""Reduce a rocprofv3 kernel trace of a llama.cpp decode loop to the per-shape
weight-bandwidth table of the decode comparison.

The reference decode graph runs one ``mul_mat_vec_q`` per projection and one
``quantize_q8_1`` per matvec to prepare its Q8_1 activation row, so the trace
carries the shape the kernel signature does not: the grid of the quantize
launch immediately before a matvec is that matvec's ``K``, and the matvec grid
is ``32 * N`` for this kernel on this device. The dispatch order therefore
pairs each matvec with its shape without reading the model, and the reduction
fails when a matvec has no preceding quantize launch.

Shapes are aggregated by ``(dtype, K, N)``, which is exactly the signature the
yuuka per-tensor attribution prints. When a yuuka table is supplied, the
reduction joins the two by that signature: the yuuka row names the projection,
and the two bandwidths become directly comparable. The join also validates that
the two engines run the same number of operations per token for each signature;
the one permitted ratio of two is llama.cpp's fused gate/up pair, one dispatch
over the concatenated matrix against yuuka's two.

Only the Python standard library is used; the rocpd database is plain SQLite.
"""

import argparse
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict

# GGML type ids reported in the demangled mul_mat_vec_q template arguments.
GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
}

# Packed row size per dtype, the same table scripts/decode_profile.py uses.
ROW_BYTES = {
    "F16": lambda n_in: n_in * 2,
    "Q4_0": lambda n_in: n_in // 32 * 18,
    "Q5_0": lambda n_in: n_in // 32 * 22,
    "Q8_0": lambda n_in: n_in // 32 * 34,
    "Q4_K": lambda n_in: n_in // 256 * 144,
    "Q5_K": lambda n_in: n_in // 256 * 176,
    "Q6_K": lambda n_in: n_in // 256 * 210,
}

MATVEC = re.compile(r"mul_mat_vec_q<\(ggml_type\)(\d+),")
QUANTIZE = "quantize_q8_1"
# The reference launches one block per 32 output rows and 32 lanes per block,
# so a dispatch covers 32 * N output rows.
BLOCKS_PER_ROW = 32
# One warp per block, so the block count of a dispatch is one per 32 rows.
YUUKA_CHANNELS_PER_BLOCK = 32
REFERENCE_ROWS_PER_BLOCK = 2


class Fail(Exception):
    """A user-visible failure: bad input, unsupported schema, or broken pairing."""


class Database:
    """A rocpd database with lazily resolved, UUID-suffixed table names."""

    def __init__(self, path):
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "**", "*_results.db"),
                                     recursive=True))
            if len(found) != 1:
                raise Fail(f"{path}: expected exactly one rocprofv3 *_results.db,"
                           f" found {len(found)}")
            path = found[0]
        if not os.path.isfile(path):
            raise Fail(f"missing rocprofv3 database: {path}")
        self.path = path
        try:
            self.conn = sqlite3.connect(path)
        except sqlite3.Error as error:
            raise Fail(f"{path}: cannot open database: {error}") from error
        self.tables = [row[0] for row in self.conn.execute(
            "select name from sqlite_master where type='table'")]

    def table(self, base):
        prefix = "rocpd_" + base + "_"
        for name in self.tables:
            if name.startswith(prefix):
                return name
        raise Fail(f"{self.path}: missing rocpd table '{base}' (unsupported schema?)")

    def dispatches(self):
        """Return every dispatch in submission order, with its launch geometry."""
        kernel_dispatch = self.table("kernel_dispatch")
        kernel_symbol = self.table("info_kernel_symbol")
        query = (
            f"select kd.start, kd.end, kd.grid_size_x, kd.grid_size_y,"
            f" ks.display_name"
            f" from {kernel_dispatch} kd"
            f" join {kernel_symbol} ks on ks.id = kd.kernel_id"
            f" order by kd.start, kd.id"
        )
        dispatches = []
        try:
            for row in self.conn.execute(query):
                dispatches.append({
                    "ns": float(row[1]) - float(row[0]),
                    "grid_x": row[2],
                    "grid_y": row[3],
                    "name": row[4] or "",
                })
        except sqlite3.Error as error:
            raise Fail(f"{self.path}: cannot read kernel dispatches: {error}") from error
        if not dispatches:
            raise Fail(f"{self.path}: no kernel dispatch; was llama-bench run under"
                       " rocprofv3?")
        return dispatches


def pair_shapes(dispatches):
    """Return matvec dispatches paired with the K of the preceding quantize."""
    pairs = []
    pending_k = None
    for dispatch in dispatches:
        if QUANTIZE in dispatch["name"]:
            pending_k = dispatch["grid_x"]
            continue
        match = MATVEC.search(dispatch["name"])
        if not match:
            continue
        if pending_k is None:
            raise Fail("a mul_mat_vec_q dispatch has no preceding quantize_q8_1;"
                       " the K/N pairing rule does not hold for this trace")
        dtype_id = int(match.group(1))
        if dtype_id not in GGML_TYPE_NAMES:
            raise Fail(f"unsupported ggml type id {dtype_id} in a mul_mat_vec_q"
                       " dispatch")
        if dispatch["grid_x"] % BLOCKS_PER_ROW:
            raise Fail(f"mul_mat_vec_q grid {dispatch['grid_x']} is not a multiple of"
                       f" {BLOCKS_PER_ROW}; the N-from-grid rule does not hold")
        pairs.append({
            "dtype": GGML_TYPE_NAMES[dtype_id],
            "n_in": int(pending_k),
            "n_out": int(dispatch["grid_x"] // BLOCKS_PER_ROW),
            "ns": dispatch["ns"],
        })
        # One quantize belongs to one matvec; do not let a stale K survive a
        # second matvec without its own quantization.
        pending_k = None
    if not pairs:
        raise Fail("no mul_mat_vec_q dispatch in the trace")
    return pairs


def aggregate(pairs, steps):
    """Return {signature: {count, ns, rows}} divided per decode step."""
    groups = defaultdict(lambda: {"count": 0, "ns": 0.0})
    for pair in pairs:
        key = (pair["dtype"], pair["n_in"], pair["n_out"])
        groups[key]["count"] += 1
        groups[key]["ns"] += pair["ns"]
    if steps <= 0:
        raise Fail("the trace holds no decode step")
    for entry in groups.values():
        entry["ops"] = entry["count"] / steps
        entry["ns_per_step"] = entry["ns"] / steps
    return groups


def count_steps(pairs):
    """Return the decode steps the trace covers, one lm_head matvec per step."""
    lm_head = [pair for pair in pairs if pair["n_out"] == 248320]
    if not lm_head:
        raise Fail("no lm_head-sized matvec (248320 output rows); the trace does not"
                   " cover a full decode step")
    return len(lm_head)


def parse_yuuka_table(path):
    """Return the per-tensor matmul rows of TestGPUProfileDecode, by signature."""
    rows = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                # t.Logf output, optionally prefixed by "<file>.go:<line>: ":
                # "<tensor> <dtype> <K>-><N>[ x2] <ops> ... <GB/s> <share>%".
                match = re.search(
                    r"(?:\S+\.go:\d+:)?\s*(\S+)\s+(\S+)\s+(\d+)->(\d+)(\s+x2)?"
                    r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+%\s*$", line)
                if not match:
                    continue
                dtype = match.group(2)
                if dtype not in ROW_BYTES:
                    continue
                n_in = int(match.group(3))
                n_out = int(match.group(4))
                paired = match.group(5) is not None
                ops = int(match.group(6))
                mbytes = float(match.group(8))
                gbps = float(match.group(9))
                key = (dtype, n_in, n_out)
                entry = rows.setdefault(key, {
                    "names": [], "ops": 0, "bytes": 0.0, "ms": 0.0, "gbps": 0.0,
                    "paired": False,
                })
                entry["names"].append(match.group(1))
                entry["ops"] += ops
                entry["bytes"] += mbytes * 1e6
                entry["ms"] += (mbytes * 1e6 / (gbps * 1e6)) if gbps else 0.0
                entry["paired"] = entry["paired"] or paired
                entry["gbps"] = (entry["bytes"] / (entry["ms"] * 1e6)
                                 if entry["ms"] else 0.0)
    except OSError as error:
        raise Fail(f"{path}: cannot read the yuuka table: {error}") from error
    if not rows:
        raise Fail(f"{path}: no per-tensor matmul row; was TestGPUProfileDecode run"
                   " with -v?")
    return rows


def format_value(value, digits=1):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_table(label, columns, rows):
    print(label)
    widths = []
    for index, name in enumerate(columns):
        width = len(name)
        for row in rows:
            width = max(width, len(str(row[index])))
        widths.append(width)
    print("  ".join(name.ljust(widths[index]) for index, name in enumerate(columns)))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index])
                        for index, value in enumerate(row)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reduce a rocprofv3 trace of a llama.cpp decode loop to the"
                    " per-shape weight-bandwidth table of the decode comparison.")
    parser.add_argument("--db", required=True,
                        help="rocprofv3 rocpd database or its directory")
    parser.add_argument("--yuuka-table", default=None,
                        help="TestGPUProfileDecode output; joins the two engines by"
                             " (dtype, K, N)")
    args = parser.parse_args(argv)

    try:
        database = Database(args.db)
        pairs = pair_shapes(database.dispatches())
        steps = count_steps(pairs)
        reference = aggregate(pairs, steps)
        print(f"database={database.path}")
        print(f"steps={steps} matvec_dispatches={len(pairs)}"
              f" matvec_dispatches_per_step={len(pairs) / steps:.0f}")
        print()

        for key, entry in reference.items():
            dtype, n_in, n_out = key
            entry["matrix_bytes"] = ROW_BYTES[dtype](n_in) * n_out

        if not args.yuuka_table:
            rows = []
            for key in sorted(reference, key=lambda k: (k[2], k[1], k[0])):
                entry = reference[key]
                gbps = (entry["matrix_bytes"] * entry["ops"]
                        / entry["ns_per_step"])
                rows.append([
                    GGML_TYPE_NAMES.get(key[0], key[0]), f"{key[1]}->{key[2]}",
                    format_value(entry["ops"], 0),
                    format_value(entry["ns_per_step"] / entry["ops"] / 1000.0, 1),
                    format_value(gbps),
                    format_value(key[2] / REFERENCE_ROWS_PER_BLOCK, 0),
                ])
            print_table("llama.cpp per-shape weight read (one matvec per projection):",
                        ["dtype", "K->N", "ops/step", "us/op", "GB/s", "blocks"],
                        rows)
            return 0

        yuuka = parse_yuuka_table(args.yuuka_table)
        missing_reference = sorted(set(yuuka) - set(reference))
        missing_yuuka = sorted(set(reference) - set(yuuka))
        if missing_reference:
            raise Fail("the reference trace has no dispatch for signature(s) "
                       + ", ".join(f"{d} {k}->{n}" for d, k, n in missing_reference))
        if missing_yuuka:
            raise Fail("the yuuka table has no row for signature(s) "
                       + ", ".join(f"{d} {k}->{n}" for d, k, n in missing_yuuka))

        rows = []
        for key in sorted(yuuka, key=lambda k: (-yuuka[k]["ms"], k[2], k[1])):
            dtype, n_in, n_out = key
            mine = yuuka[key]
            theirs = reference[key]
            ratio_ops = theirs["ops"] / mine["ops"] if mine["ops"] else 0.0
            if abs(ratio_ops - 1.0) > 1e-6 and abs(ratio_ops - 2.0) > 1e-6:
                raise Fail(f"{dtype} {n_in}->{n_out}: the reference runs"
                           f" {theirs['ops']:.0f} operations per step and yuuka"
                           f" {mine['ops']:.0f}; the two engines do not partition this"
                           " projection the same way, so the bandwidths are not"
                           " comparable")
            # The fused gate/up dispatch covers two matrices; scale the matrix
            # bytes by the operation ratio, which the check above bounds to 1 or 2.
            reference_bytes = mine["bytes"] * ratio_ops
            ref_gbps = reference_bytes / theirs["ns_per_step"]
            rows.append([
                ",".join(sorted(set(mine["names"]))), dtype,
                f"{n_in}->{n_out}" + (" x2" if mine["paired"] else ""),
                format_value(-(-n_out // YUUKA_CHANNELS_PER_BLOCK), 0)
                if n_out >= YUUKA_CHANNELS_PER_BLOCK else "1",
                format_value(mine["gbps"]),
                format_value(n_out / REFERENCE_ROWS_PER_BLOCK, 0),
                format_value(ref_gbps),
                format_value(ref_gbps / mine["gbps"], 2) if mine["gbps"] else "-",
            ])
        print()
        print_table(
            "per-shape weight bandwidth inside one decode step"
            " (yuuka: TestGPUProfileDecode, reference: this trace):",
            ["tensor", "dtype", "K->N", "yuuka blocks", "yuuka GB/s",
             "ref blocks", "ref GB/s", "ref/yuuka"],
            rows)
    except Fail as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

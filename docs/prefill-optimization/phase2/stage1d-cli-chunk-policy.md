# Stage 1d — Select the CLI prefill chunk from measured prompt lengths

Execution order 5. Start the sweep on a revision with accepted 1a, 1c and both parts of 1b. Record their SHAs. Sub-128 projection arithmetic affects the cost and outputs of the last chunk. The measurements at `9da6e77` and `260094f` identify boundary cases but cannot select a new default. See the [roadmap](roadmap.md) and [sub-128 plan](stage1bc-sub128-rows.md).

## Iteration 1 — Measure the choice before writing the rule

1. Build a Release HIP CLI and the [matched-token driver](../../../bench/prefill-compare/main.go) against the same library. Use the roadmap GGUF/hash, exclusive gfx1151, greedy, concurrency one. Record ROCm/HIP, kernel, power and temperatures. Generate one 8192-ID file with the driver's `-write-tokens` and reuse prefixes. Run each cell at chunk 512, 1024 and 2048. Use 2 warmups and 5 repetitions per process. Interleave 512/candidate/candidate/512 to confirm each proposed crossover.
2. Measure at 512, 513, 530, 600, 1024, 1025, 1042, 2048, 2049, 2066, 4096, 4097, 4114 and 8192. Include 127/128/129 as short controls. For each cell record actual chunk sequence, unprofiled median to host logits, spread, top-1 and max absolute logit difference. A 2049-token prompt at chunk 2048 still has a one-row tail. Time the entire prompt.
3. The driver fixes F16 KV and caps the chunk at prompt length. Separately exercise the [CLI](../../../cmd/yuuka/main.go) using the same input per A/B, rendered chat and raw text, including its default F32 KV. Record actual token counts, engine TTFT, elapsed time to the first emitted token, and prefill alone. Cover Q8_0 explicitly before enabling a policy there. Use a fixed context large enough for prompt and continuation in each comparison. CLI setup may reserve a larger buffer than the driver.
4. Use the byte sizes in [configurePrefill](../../../engine/gpu/model.go) to sum the live prefill allocations, including split/BLAS scratch when allocated. Record peak device allocation during setup alongside that sum and the explicit-chunk control at the same capacity/KV. The published F16 figures at chunks 512/2048 were 191,924,224/767,696,896 prefill bytes for the recorded long-context configuration. For this sweep, use the largest in-scope explicit setting (2048) at the same model, capacity and KV format as the upper bound on prefill bytes, subject to any lower deployment limit. Require the proposed automatic choice to allocate no more than its explicit equivalent. Report the increase over 512 in MiB for each promoted band. [Resource baseline](../phase1/acceptance/07-long-attention.md)

For example, after building the binaries and setting `MODEL` and `COMPARE`, run one F16 boundary pair as follows. Repeat for the full matrix and interleave process order for confirmation.

```bash
out=$(mktemp -d)
"$COMPARE" -model "$MODEL" -write-tokens -tokens "$out/ids" -prompt 8192
for chunk in 512 2048 2048 512; do
  "$COMPARE" -model "$MODEL" -tokens "$out/ids" -prompt 2049 -chunk "$chunk" \
    -warmup 2 -reps 5 -logits "$out/logits-$chunk"
done
```

## Iteration 2 — Apply the smallest passing rule

In `run`, resolve the chunk after `RenderPrompt` and `EncodePrompt`, using `len(ids)`. In `bench`, resolve after truncating the corpus to `promptTokens`. Share a small CLI-local selector and use `FlagSet.Visit` to detect an explicit `chunk-size`, including an explicitly supplied 512. Pass the result once to `buildEngine`. Keep explicit-value validation and [runtime clamping](../../../engine/gpu/runtime.go). Limit automatic selection to validated HIP model/KV configurations. Keep 512 for uncovered cells. Use at most the length breakpoints actually supported by data. Exercise both CLI entry points and add threshold-neighbor and override cases to the [CLI tests](../../../cmd/yuuka/main_test.go). Server scheduling has its own chunk policy.

## Decision and acceptance

- Select a length band only if the paired median TTFT improves across its aligned and boundary cases in repeat runs. Require a net reduction across the matrix (report the equal-weight mean of per-cell TTFT ratios), with no reproducible boundary slowdown outside measured control-to-control noise. If equally fast, choose the smaller allocation. Publish the per-cell results even when 512 wins throughout.
- With the rule active, the omitted flag must select the intended explicit-chunk route, with identical logits and greedy tokens for the same IDs and actual chunking. Across different chunkings check top-1 and three continuation tokens, report full-vocabulary max difference, and pass the existing 1024-token chunk-invariance test. The roadmap records a pre-existing 1.0674 difference for 513 split vs unsplit. Never use this as a new same-chunk tolerance.
- Verify explicit 512/1024/2048, explicit invalid HIP sizes, capacity clamping, default F32, CPU behavior and chat-rendered lengths. Run `go test ./cmd/yuuka`, the HIP long-context tests, and smoke both `yuuka run` and `yuuka bench` on the reference device. Record memory, paired samples and the selected intervals in Phase 2 acceptance before changing the default.

Estimated production change if a rule passes: 25–55 Go lines in one CLI file, plus table tests and a measurement record. A sweep that finds no passing band needs only the record.

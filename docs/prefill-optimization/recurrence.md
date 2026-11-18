# Gated delta-net recurrence

The 24 recurrent layers spend 53.9 ms at pp512 versus 35.1 ms upstream.
The split-register core accounts for 45.7 vs 13.2 ms, and the surrounding
convolution, preparation and gating for 8.2 vs 21.9 ms. Yuuka has room to
win on the complete operation if it preserves its inexpensive preparation.
The proposed budget is **30 ms** for all 24 layers. At the baseline support
cost this leaves about 22 ms for the core including any new preparation.

The upstream trace on this device executed a per-token kernel with eight
lanes per column. At baseline the two cores issued about 9.50M vs 3.15M
LDS instructions per layer. Two experiments isolate the cost of repeated
preparation from the cost of mapping the state update to lanes.
[Upstream implementation at v1.0.0](https://github.com/ggml-org/llama.cpp/blob/b1b770d90d94cad9e9ea5b993bdbc383a522fb28/ggml/src/ggml-cuda/gated_delta_net.cu#L39-L119)

## Step 1. Prepare once per token and key group

1. In `launch_recurrent_chunk_f32` keep the current order of
   `conv1d_chunk_f32_kernel` and `conv1d_state_f32_kernel`. For the candidate
   128-state route, insert one preparation launch, run a prepared-input
   version of the four-lane core, then run `gdn_norm_gate_rows_f32_kernel`
   as today. The existing core normalizes the same Q/K once per head and
   column workgroup and executes two block barriers per token.
   [Launcher](../../kernels/hip/prefill/prefill.hip#L825) · [Current core](../../kernels/hip/prefill/prefill.hip#L392)
2. Launch 128 threads per `(token r, key group g)`. Give thread `d` the
   Q and K components at `g*128+d` and `groups*128+g*128+d` in convolution
   row `r`. For each norm use the current Wave32 sum of squares and sum
   the four wave partials in order. Write FP32
   `Q[d] = (q_raw*q_scale)*rsqrtf(q_sum+eps)` and
   `K[d] = k_raw*rsqrtf(k_sum+eps)`. The group also prepares each head
   `h = g, g+groups, ... < heads` exactly once. With
   `x = alpha[r,h]+dt[h]`, use `softplus = x > 20 ? x : log1pf(expf(x))`,
   `decay = expf(softplus*a[h])` and `b = 1/(1+expf(-beta[r,h]))`.
   The existing triangular preparer stores **log** decay and uses a
   different Q reduction order. The per-token core consumes the actual
   decay factor. [Triangular preparer](../../kernels/hip/prefill/gdn_chunk.hip#L83)
3. Lay out one FP32 span as consecutive arrays
   `Q[rows,groups,128]`, `K[rows,groups,128]`, `decay[rows,heads]`,
   `b[rows,heads]`. In `addPrefillRecurrent` bind an optional prepared
   buffer to the unused `RECURRENT_CHUNK.buf[13]`. On the target graph
   use `p.ffnGate` if its byte capacity covers
   `4*rows*(2*groups*128+2*heads)`. Its previous layer has finished,
   and this layer's first FFN write follows the entire recurrent op on
   the same stream. The buffer can be reused after the core's last read.
   Keep `buf[11]` for raw convolution (including V) and `buf[12]` for
   the triangular route. Accept at most one of slots 12 and 13. Check
   slot 13 independently with checked products and byte span in
   `validate_recurrent_chunk_plan_op` because slot 12 can be null.
   Carry it through `RecurrentChunkArgs` and the native launcher. For
   other geometries or insufficient spare capacity select the current
   route at planning time. Reject a malformed bound span during validation.
   [Allocation](../../engine/gpu/model.go#L379) · [Plan binding](../../engine/gpu/plan.go#L616) · [Go descriptor](../../backend/hip/hip.go#L1541) · [Native validation](../../backend/hip/bridge/bridge.cpp#L614)
4. Keep the baseline four-lane dot and update order. Read the prepared
   Q/K for `g = h % groups` and both scalar gates directly in each token
   iteration. Remove the shared Q/K arrays, scalar handoff and both
   block barriers from **this core**. Read V from `conv[r,2*groups*128+h*128+col]`.
   Measure the additional kernel and global reads together with the core.

## Step 2. Change only the column mapping

Keep the prepared inputs and 128 threads per block. At four lanes per
column a block owns 32 columns with 32 state rows per lane. In the proposed
eight-lane version a block owns 16 columns with 16 state rows per lane.
Launch `grid=(heads,8)` for 128 columns. For thread `tid`, use
`col = 16*blockIdx.y + tid/8`, `part = tid%8` and
`stateRow = 16*part+j` for `j=0..15`. Each head's matrix remains
row-major. Load `s[j] = matrix[stateRow*128+col]` once before the token
loop. Each aligned eight-lane group stays inside a Wave32.

```text
for each token in order:
    for j ascending: s[j] *= decay, accumulate sk = fmaf(s[j], K[stateRow], sk)
    sum sk across the eight lanes with XOR distances 1, 2, 4
    delta = (V[col] - sk) * b
    for j ascending: s[j] = fmaf(K[stateRow], delta, s[j])
                     accumulate y = fmaf(s[j], Q[stateRow], y)
    sum y across the eight lanes with XOR distances 1, 2, 4
    part zero writes raw out[r,h,col]
store each s[j] back to matrix[stateRow*128+col] after the last token
```

Initialize `sk` and `y` to zero for each token. The change halves state
elements per lane and doubles column workgroups per head. It changes FP32
reduction order. Check final state, VGPRs, spills and the preparation-plus-core
time before selecting either lane count. Keep both convolution history and
matrix state in their current session-owned buffers.

## Verify and decide

- Extend [test_recurrent_chunk](../../tests/hip/prefill_test.cpp#L1424) to compare
  convolution history, raw recurrence outputs, final matrix state and gated
  outputs of both variants with the CPU reference. Include a nonzero
  initial state, heads sharing a key group and a sequence split into two
  chunks. Check repeatability and the current operator tolerances.
- Run real-model chunked prefill, greedy continuation and the separate
  decode-path check. Inspect layer state at the chunk boundary if the first
  token agrees and subsequent tokens diverge. [CPU recurrence](../../model/qwen35/recurrent.go)
  · [CPU/GPU comparison tests](../../engine/gpu/runtime_test.go)
- Time four-lane baseline, four-lane with prepared inputs, and eight-lane
  with the same inputs. Include convolution, preparation, core, output gate
  and total. An intermediate milestone is **42 ms** at pp512. Revisit
  the 30 ms budget against the end-to-end result. Retain the existing path
  for a geometry where the experiment loses.

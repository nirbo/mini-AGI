# Ordinary FP8 expert training with Gefen on RTX 5090

This experiment evaluates native FP8 expert matrix products while Gefen updates
FP32 master weights. AdamW handles the other model parameters. Muon is outside
this implementation. The research adapter is isolated from the trainer's public
backend choices and does not change checkpoint formats or precision defaults.

## Arithmetic and cache ownership

`fp8_experts.py` implements three independent expert projections, with these
policies:

- Forward matrix operands use ordinary FP8 E4M3 with per-row FP32 scales.
  Activation rows are individual tokens, so their scales do not depend on
  future tokens. Weight scales are local to each expert and output row.
- Backward signals use E5M2; the weight/input operands use E4M3. All three
  linear GEMMs (forward, input gradient, weight gradient) consume native FP8.
  FP32 GEMM output is rescaled to BF16 activations or FP32 master gradients.
- SwiGLU is BF16. Routing, admission, gates, scores and the remaining model
  computation retain their existing policies. The optimizer updates FP32
  weights; W8A8 describes matrix operands, not optimizer state or every tensor.
- Forward and backward weight orientations are quantized independently from
  the masters. Detached data and scales are cached by owner identity, storage,
  tensor version, optimizer generation, paging generation and slot identities.
  Both orientations are retained, so the weight cache is not simply half the
  size of an existing one-orientation BF16 cache.
- Quantized generations bind before activation checkpointing. Old closures
  retain their original quantized data while the current cache is replaced.
  Every use has an independent FP32 master-gradient edge. Cached data created
  during inference remain usable when training resumes.

This is ordinary FP8, not MXFP8: stored scales are FP32 values per row, with no
E8M0 scale per 32-element block. QuACK can select SM120's block-scaled MMA
instruction with constant unit scales as an internal fast path for plain FP8.
That instruction choice does not change the stored quantization format.

QuACK provides batched FP8 GEMMs; no custom MMA kernel is added. Its public
`gemm(..., tuned=False)` selects the existing architecture configuration.
The prototype materializes FP32 GEMM output and runs a separate compiled
rescaling operation. Dispatch pads every expert to the largest accepted
assignment count, rounded to 16; empty rows remain zero. These are explicit
optimization costs, particularly against Sonic's fused sparse BF16 path.

The custom backward is an approximate quantized-training estimator. It is not
the mathematical derivative of rounding. Tests check each actual native GEMM
against decoded operands, then compare expert outputs and all derivatives
against an independent BF16 assignment-by-assignment reference.

## Qualification and measurement protocol

The repository's existing `.venv` and installed QuACK/Sonic/Gefen stack are
used. No packages are installed or upgraded, and no GPU/system settings change.
GPU processes run serially under external read-only NVML monitoring, including
JIT, warmup, evaluation and calibration. Timed windows have explicit monotonic
timestamps. Sensor averaging, sampled peaks and desktop activity limit the
precision of board energy; CPU/system energy is excluded.

CUDA qualification covers native FP8 forward/dgrad/wgrad, FP32 gradients,
zero/negative scores, unused experts, missing tokens, causal prefixes with
capacity limits and stochastic routing, cache hits/misses, inference-to-training
reuse, paging, checkpoint replay, and actual Gefen updates. Fixed-route expert
output relative L2 must be below 0.12 and every derivative below 0.18 versus
BF16; native decoded-operand GEMMs must be below 0.005. Exact cache/replay and
causal-prefix assertions use zero tolerance where stated by the tests.

The real-checkpoint screen opens the downloaded undertrained weights read-only:
hidden 512, expert width 2048, 32 resident experts, fixed depth 6, context 512,
gradient clipping at norm 1. Every touched expert remains cached in host RAM;
learned weights and optimizer history stay in RAM. No trained checkpoint is
saved. Hashes of all original checkpoint files are checked before/after.

Both paths start with fresh Gefen/auxiliary optimizer history. Expert LR is
`3e-5`, auxiliary pool LR `1e-4`, trunk LR `2e-5`, betas `(0.9,0.95)`, weight
decay 0.1. Rates are held fixed across precision paths; this experiment does
not search for an FP8-specific optimum. There are 256 updates on each of seeds
42, 43 and 44, with identical window permutations within a BF16/FP8 pair.

TinyStories validation text supplies disjoint token ranges: adaptation before
262144, tuning from 262144, test from 524288. Each evaluation uses sixteen
512-token windows. These byte/marker-token ranges are not a canonical
document-deduplicated split; original pretraining overlap is unknown. Pool
selection temperature stays zero; these are data-order seeds.

An eight-update gate repeats each precision in independent processes and
requires exact initial weights, first gradients, final weights/history/host
experts, slots, objectives and evaluations within that precision. FP8 also
requires exact cached/uncached results. BF16 and FP8 evaluations can differ
because their arithmetic differs. Each seed-42 long run checks its eight-step
objective/gradient-norm prefix against the corresponding replay oracle.
The whole 256-update trajectory is not independently replayed.

Training measurements include both the complete 256-update interval and the
predeclared interval after 16 updates. The latter still includes actual paging,
state transfers and any late compilation/calibration. It is not an assertion
that every initialization cost has disappeared. Source hashes include the
executed numerical code, model code, loader and configuration.

Synthetic linear probes measure complete forward/backward at 128/1024 rows
and widths 512/2048, plus a 4096-square case. Torch native scaled GEMM and
QuACK are compared with cached/uncached conversions. BF16 includes its own
cached cast reference with independent FP32 gradient edges. Each linear probe
also records output/input-gradient/weight-gradient errors versus BF16.

Expert-layer probes compare padded BF16, fused Sonic BF16 and padded FP8, each
with cached/uncached weights. Current geometry is 4096 input tokens, 32 experts,
top-8, hidden 512, width 2048. Large geometry is 2048 tokens, four experts,
top-2, hidden 4096, width 16384. These synthetic assignments are balanced;
real routing can produce substantially different padding/work. Large expert
probes require finite derivatives, not demonstrated convergence at that size.

Timing uses three seconds after warmup and a 1.1-second sensor/workload settling
period, repeated twice with reversed method/shape order. Benchmark and model
processes use a 75% allocator cap. Synthetic fixtures, cached data and linear reference
buffers contribute to reported peak allocation; these are not complete-model
training-fit measurements. Only real model runs report language-model new
tokens/s and J/new token. Synthetic runs report input rows/tokens and J/row.

## Commands

Run from the repository root. No installation is required in the qualified
local environment. To repeat the standalone qualification:

```sh
PATH="$PWD/.venv/bin:$PATH" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
MINAGI_FP8_TEST_CUDA=1 MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH=1 \
TORCHINDUCTOR_COMPILE_THREADS=2 \
.venv/bin/python -m replication.gpu_monitor \
  --out runs/fp8-tests.power.json -- \
  .venv/bin/python -m unittest tests.test_fp8_experts -v
```

The quantization/rescale compiled functions use an explicit per-region
specialization limit of 64 on the qualified Torch version. This accommodates
the 2D/3D and dtype/autocast cases without changing system configuration.

```sh
.venv/bin/python -m replication.run_fp8_screen --phase repro
.venv/bin/python -m replication.run_fp8_screen --phase probes
.venv/bin/python -m replication.run_fp8_screen --phase learning
.venv/bin/python -m replication.summarize_fp8_screen
```

The runner supplies external telemetry and the local Gefen build opt-in. Cached
learning results are reused only when configuration and source hashes match.
Timing probes require fresh outputs. Use `--root` consistently across phases
for a separate experiment location.

For one directly monitored dense probe, including conversion and backward:

```sh
PATH="$PWD/.venv/bin:$PATH" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TORCHINDUCTOR_COMPILE_THREADS=2 \
.venv/bin/python -m replication.gpu_monitor \
  --out runs/fp8-linear-large.power.json -- \
  .venv/bin/python -m replication.probe_fp8 --phase linear --large \
  --seconds 3 --out runs/fp8-linear-large.json
```

This is an experimental process-local adapter. There is deliberately no
`train.py --precision fp8` option or production FP8 checkpoint format yet.

## Results

Results are populated after the complete measured screen and final checks.

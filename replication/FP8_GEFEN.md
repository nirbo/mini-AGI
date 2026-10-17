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

The prototype passes its arithmetic/cache/causality qualification, but the current model does not justify switching from Sonic BF16. Large cached synthetic layers demonstrate a useful FP8 regime; they do not establish full-model convergence or fit at that width.

### Matched checkpoint adaptation

| Data-order seed | Sonic BF16 test CE | Ordinary FP8 test CE | FP8 minus BF16 |
| --- | ---: | ---: | ---: |
| 42 | 0.574349 | 0.646239 | +0.071890 |
| 43 | 0.648714 | 0.655389 | +0.006674 |
| 44 | 0.589972 | 0.592555 | +0.002584 |

Mean final test CE: BF16 **0.604345**, FP8 **0.631394**. Initial CE is 1.455013 / 1.453381, respectively. Evaluation uses each path's own arithmetic. This is a short fixed-rate screen, without an FP8-specific rate search or a common-arithmetic evaluation of the learned weights.

| Training interval | Backend | Seconds | New tokens/s | Avg / peak W | J/new token | Peak allocated GiB¹ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| All 256 updates | Sonic BF16 | 46.65 | 2810 | 174 / 196 | 0.0619 | 1.64 |
| All 256 updates | Ordinary FP8 | 51.63 | 2539 | 200 / 237 | 0.0786 | 1.99 |
| Updates 17–256 | Sonic BF16 | 33.75 | 3641 | 181 / 196 | 0.0496 | 1.64 |
| Updates 17–256 | Ordinary FP8 | 38.20 | 3217 | 212 / 237 | 0.0661 | 1.99 |

Values are means over three seeds; peaks are maxima. ¹Memory is the peak over the complete training interval, repeated in both rows. Loading and held-out evaluation are outside these training intervals. The complete interval includes optimizer startup/calibration; updates 17–256 can still include paging and late initialization.

### Synthetic forward/backward costs

Each timing cell spans the two independently ordered repeats. Rows/s are total completed rows divided by the window wall time (not the reciprocal of median iteration time). Rows/s and power/energy are means of the two windows; peak W and allocation are maxima. Cached cases keep the same weights throughout the synthetic loop. There is no optimizer in these probes.

| Linear M×K×N | Backend | Cached | Time ms | Input rows/s | Avg / peak W | mJ/input row | Peak GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024×512×2048 | BF16 | yes | 0.067–0.075 | 11877616 | 354 / 397 | 0.030 | 0.10 |
| 1024×512×2048 | Torch FP8 | yes | 0.196–0.401 | 3858700 | 178 / 209 | 0.047 | 0.11 |
| 1024×512×2048 | QuACK FP8 | yes | 0.476–0.482 | 2181631 | 152 / 178 | 0.070 | 0.11 |
| 4096×4096×4096 | BF16 | yes | 2.126–2.137 | 1919074 | 601 / 606 | 0.313 | 0.62 |
| 4096×4096×4096 | Torch FP8 | yes | 1.750–1.753 | 2335181 | 575 / 577 | 0.246 | 0.73 |
| 4096×4096×4096 | QuACK FP8 | yes | 1.552–1.554 | 2628003 | 582 / 592 | 0.222 | 0.80 |

| Expert geometry | Backend | Cached | Time ms | Input tokens/s | Avg / peak W | mJ/input token | Peak GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 2048 | Padded BF16 | no | 8.732–8.734 | 468355 | 566 / 573 | 1.210 | 1.49 |
| 512 / 2048 | Padded BF16 | yes | 8.276–8.294 | 493331 | 576 / 579 | 1.167 | 1.49 |
| 512 / 2048 | Sonic BF16 | no | 5.431–5.444 | 752463 | 560 / 564 | 0.744 | 1.35 |
| 512 / 2048 | Sonic BF16 | yes | 5.022–5.026 | 814736 | 572 / 575 | 0.702 | 1.47 |
| 512 / 2048 | Padded FP8 | no | 12.660–12.688 | 322855 | 522 / 527 | 1.617 | 1.84 |
| 512 / 2048 | Padded FP8 | yes | 10.946–10.967 | 373504 | 519 / 524 | 1.389 | 1.84 |
| 4096 / 16384 | Padded BF16 | no | 44.956–45.066 | 45459 | 587 / 590 | 12.921 | 7.19 |
| 4096 / 16384 | Padded BF16 | yes | 41.030–41.126 | 49808 | 599 / 601 | 12.017 | 8.69 |
| 4096 / 16384 | Sonic BF16 | no | 36.304–36.310 | 56340 | 579 / 581 | 10.272 | 7.19 |
| 4096 / 16384 | Sonic BF16 | yes | 32.282–32.286 | 63359 | 598 / 600 | 9.439 | 8.69 |
| 4096 / 16384 | Padded FP8 | no | 41.424–41.446 | 49369 | 537 / 541 | 10.874 | 9.17 |
| 4096 / 16384 | Padded FP8 | yes | 23.247–23.295 | 87905 | 558 / 561 | 6.348 | 9.17 |

Geometry labels are hidden / expert width. The large layer has 805,306,368 expert parameters, four experts and top-2 over 2048 input tokens. Reported memory covers this layer fixture and its FP32 gradients, caches and activations; it does not include a complete recurrent model or optimizer state. The 75% allocator cap is not evidence that a full model at this width will fit.

Cached large-layer FP8 is **1.39× faster** than cached Sonic BF16. Without caching it is slower than Sonic. At the current expert width, the FP8 prototype remains slower even with caching. These results isolate the importance of conversion amortization while retaining FP32 gradients throughout.

### Conversion and gradient attribution

FP8 GEMM inputs already include FP8 backward signals. Master `.grad` buffers are FP32 and are newly computed each update. A buffer can be allocated once, but its values must be recomputed; changing its storage dtype cannot make a single quantization valid for the entire run. Weight caches remain valid only until their master generation changes.

| Dense diagnostic | Cached | Quantize ms/step | FP8 GEMMs ms/step | Rescale ms/step | Other kernels + copies ms/step | Quantizations/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024×512×2048 | no | 0.017 | 0.032 | 0.012 | 0.053 | 6 |
| 1024×512×2048 | yes | 0.011 | 0.032 | 0.012 | 0.041 | 4 |
| 4096×4096×4096 | no | 0.273 | 0.574 | 0.242 | 0.726 | 6 |
| 4096×4096×4096 | yes | 0.175 | 0.573 | 0.242 | 0.540 | 4 |

These sums use actual kernel/copy events from the Chrome trace, excluding duplicate GPU annotation ranges. They include profiling overhead and are kernel attribution, not wall-clock throughput. Other kernels include fills and elementwise copies. Cached weights remove two conversions per linear call; the remaining four quantize the current input and current backward signal in the two required orientations. The dense fixture uses FP32 inputs/gradients; model expert activation buffers use BF16. Separate rescaling, transposition/copying, dispatch padding, and small-kernel launch overhead remain optimization candidates. These dense traces do not isolate full-model FP32 gradient accumulation or Gefen cost, so they cannot establish gradient storage as the model bottleneck.

Keeping gradients permanently in FP8 would require a new accumulation and optimizer interface, including scale handling and numerical qualification. It is a separate experiment, not a one-time initialization switch. TorchAO likewise describes quantized training as freshly quantized forward/backward GEMM operands with higher-precision results ([official workflow](https://docs.pytorch.org/ao/stable/workflows/training.html)).

To repeat the attribution profile under external power monitoring:

```sh
PATH="$PWD/.venv/bin:$PATH" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TORCHINDUCTOR_COMPILE_THREADS=2 \
.venv/bin/python -m replication.gpu_monitor \
  --out runs/fp8-profile.power.json -- \
  .venv/bin/python -m replication.profile_fp8 --out runs/fp8-profile.json
```

### Verification and evidence

The final full suite passed all **119 tests** without skips; the standalone FP8 qualification passed all **10 tests**, including eight CUDA tests. The replay gate includes two independent eight-update runs per precision and one uncached FP8 replay. Six 256-update learning runs and 84 timing cases completed. All 401 original checkpoint files retain their original SHA256 hashes. No packages, system configuration, checkpoint formats or production precision defaults were changed.

Machine-readable results: [summary](results/fp8_gefen.json), [verification/environment](results/fp8_gefen_verification.json), and [raw evidence archive](results/fp8_gefen_evidence.tar.gz). The archive contains timestamped NVML samples, learning traces, Chrome profiles, source snapshots, package source hashes and explicitly labeled excluded pilots. Sensor averaging and desktop activity affect power; CPU energy is not measured.

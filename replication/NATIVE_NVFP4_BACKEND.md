# Native NVFP4 expert backend qualification

Experimental work on `feature/native-nvfp4-experts`, based on `dev` at
`33f5591c68958ae1a99ef5f513c20ab985db1e11`. The default backend remains
`reference`. This experiment is explicitly selected with
`MINAGI_EXPERT_BACKEND=nvfp4`; it is not a recommended training path. Native
arithmetic passes qualification on the tested inputs, but performance and
learning results do not support integration into `dev` yet. The feature branch
is pushed and remains unmerged.

## Arithmetic and ownership

`minagi/nvfp4.py` implements native W4A4 forward, input-gradient and
weight-gradient GEMMs using the already installed Quartet-II quantizers and
FlashInfer/CUTLASS SM120 grouped kernels. It preserves the expert interface:
routing, causal admission, gates, resident slot identities, FP32 master weights
and Adam state stay with mini-AGI. No quantized weights or calibration state are
written to checkpoints. There is no fallback to BF16 backward GEMMs.

Forward activations use a detached power-of-two normalization for **each
token**, followed by Quartet four-over-six quantization with a fixed normalized
maximum of 2. Undoing normalization after the GEMM restores the row's scale.
This is a token-local dynamic scale, not a calibrated clipping bound. Appending
future tokens or changing another sequence cannot change it. Weight maxima are
computed separately for each expert, so future resident admissions cannot
change an existing expert's representation either.

The normalization derivative treats its scale as constant. Backward follows
Quartet's randomized Hadamard/EDEN estimator on these normalized operands;
the surrounding scale multiplications implement the corresponding chain rule.
This is a stochastic approximate gradient, not the mathematical derivative of
discrete FP4 rounding. FP32 masters remain necessary.

Experts with accepted assignments are padded to a positive multiple of 128.
Empty experts are omitted, receive zero gradients, and never create zero-K
native GEMMs. Real zero-score routes are retained because their score
derivatives can be nonzero. Forward and dgrad are grouped; wgrad is grouped
into buckets with equal padded token counts because the installed
[FlashInfer API](https://docs.flashinfer.ai/generated/flashinfer.gemm.group_gemm_nvfp4_nt_groupwise.html)
accepts a common K per call. Segmented SFA storage uses the kernel's required
physical offsets, including inter-expert gaps. Unequal per-expert alpha values
are exercised explicitly.

Activation forward quantization is batched across the 128-aligned expert
segments; its normalized bound is constant, so this preserves each segment's
quantized representation exactly. Per-expert weight maxima are reduced in one
operation. Scale packing uses concatenation and vectorized alpha multiplication.
Weight quantization and backward transforms still use per-expert eager calls.
No differentiable packed-weight cache is shared across recurrent uses. This
implementation establishes a reference for subsequent fusion work; grouped
GEMM speed alone does not predict its complete-layer cost.

## Verification

The dedicated tests cover independent decoded-operand checks for all three
native GEMMs, both actual checkpoint projection widths (512/2048), unequal
expert scales, ragged token counts, zero input/weights/gradients, unused experts,
zero-score derivatives, repeated uses, checkpoint replay, recurrent causal
prefixes in training/evaluation, capacity limits, batch isolation, compute
weight caching, paging and per-expert optimizer moments.

The estimator probe uses 64 independent samples at 512→2048 with two experts
and 128/256 padded token counts. Its reference uses independently decoded
forward operands and the normalization chain rule. A finite Monte Carlo test
can measure noise/bias at those inputs; it cannot prove unbiasedness in general.

The learning probe compares identical initialization, batches and AdamW settings
for two linear experts with different input magnitudes, plus BF16-trained
weights deployed through the same FP4 path. This is a synthetic regression
control, not evidence about language-model convergence. A reduction from the
initial error is insufficient to establish an advantage over post-training
quantization.

**Final source: `1ff7b73`. All 88 tests pass**, with optional Sonic and FP4 CUDA
tests enabled. The dedicated native tests include independent FP32 decoding of
the packed E2M1 operands and 128x4 FP8 scale layout. Each captured native GEMM
must be finite and have relative L2 error ≤0.005 against multiplication of
those same decoded operands. This excludes the much larger quantization error
against the original BF16/FP32 values.

The batching optimization is separately compared with the qualified eager
implementation at `c4bb969`. Eight configurations exercise FP32/BF16 residuals
and weights, checkpointing on/off, repeated uses, 4/32 resident slots, unused
experts and ragged assignments. Outputs, input/weight/score gradients and final
CPU/CUDA RNG states are bit-identical. This equivalence comparison uses matching
residual/score dtypes, as supported by the earlier implementation. A separate
regression covers the fixed BF16-residual/FP32-score mixing bug found during
qualification. Existing reference/Sonic arithmetic functions have identical ASTs
to `dev`; pool, paging and precision modules are unchanged.

At 64 samples the stochastic estimator's relative L2 errors against the
decoded-forward straight-through reference are **0.01726 dgrad / 0.01793 wgrad**.
At one sample they are approximately 0.138 / 0.143. Averaging estimates the
noise/bias at these particular inputs; ordinary training uses one sample.

The matched 400-update regression gives the following heldout errors. Both
experts use the same optimizer schedule and batches across backends; input
magnitudes differ by 16× between experts.

| Training and deployment | Heldout MSE |
| --- | ---: |
| BF16 training, BF16 deployment | 0.00011854 |
| BF16 training, this same NVFP4 deployment path (PTQ control) | 0.122291 |
| Native NVFP4 training and deployment | 0.169048 |

Native training reduces its initial MSE of about 16.21 by 98.96%, but finishes
**38.2% worse than the PTQ control**. This untuned experiment demonstrates
learning; it does not demonstrate a QAT advantage or language-model quality.

## Complete-layer performance

RTX 5090, 32 resident slots (one unused), 4096 input token rows, top-8 accepted
routes, hidden 512 and expert width 2048. The complete accepted-route layer
includes packing, quantization/transforms, mixture scores and all backward
gradients, starting from FP32 weight slabs without the pool's shared BF16 cast
cache. Warmup and 1.1 seconds of the same workload precede each 3-second window.
Reference/FP4/FP4/reference and Sonic/FP4/FP4/Sonic ordering controls drift.

| Backend | Median pass time | Window average W | Sampled peak W | Board J/input row | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference BF16 | 8.18–8.19 ms | 505–517 | 514–520 | 0.00101–0.00103 | 1.44–1.51 |
| Sonic BF16 | 4.27–4.28 ms | 594–597 | 601 | 0.000621–0.000623 | 1.43 |
| Native NVFP4, batched preparation | 24.99–25.38 ms | 311–327 | 322–350 | 0.00199–0.00204 | 2.08 |

The eager FP4 baseline took 27.79–27.96 ms. Arithmetic-preserving batching
reduces pass time by roughly 9–11%; it does not close the gap to either BF16 control.
Power figures are measured board telemetry, not TDP-based estimates. Layer rows
are synthetic independent inputs, not the overlapping context/new-token count
used in the model benchmark below.

## Actual recurrent-model training

The downloaded undertrained checkpoint is opened read-only. All updates use
fresh in-memory AdamW state and are discarded. BF16 remains the model compute
dtype outside the pooled FP4 expert projections; master weights remain FP32.
Context 4096, stride 2048, fixed depth 24, two warmups and eight measured updates
per process. Each checkpoint setting uses Sonic/FP4/FP4/Sonic ordering. These are
short performance checks, not a convergence experiment. Window throughput
includes work between updates; peak allocated VRAM includes warmup.

| Expert checkpointing | Backend | New tokens/s, aggregate | Average / peak sampled W | Board J/new token | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| On | Sonic BF16 | 6256 | 281 / 302 | 0.0449 | 4.24–4.26 |
| On | Native NVFP4 | 2109 | 233 / 251 | 0.1106 | 5.37–5.38 |
| Off | Sonic BF16 | 6791 | 263 / 300 | 0.0388 | 10.66–10.70 |
| Off | Native NVFP4 | 2853 | 233 / 248 | 0.0816 | 13.22–13.31 |

All losses and gradient norms are finite. Relative to Sonic, FP4 delivers only
**33.7% / 42.0% of the throughput** and costs **2.46× / 2.11× the board energy
per new token**, with checkpointing on/off respectively. It also uses more
allocated VRAM in this implementation. Its lower board power does not imply
that performance will scale proportionally toward the 600 W limit.

Sampling intervals bracket every model timing window, with maximum sample gaps
below 0.102 seconds. GPU utilization, clocks, temperature and limiting reason
masks are retained for each window. Model-run median SM clocks are about
2.93–2.94 GHz, and median temperatures are 54–59°C; no thermal limiting mask is
observed. One Sonic run includes the software-power-cap reason. Sensor averaging
and the short windows limit precise power comparisons, but cannot explain the
large throughput and energy differences.

The **eager, pre-batching** model trace has 66,895 kernel launches in one
instrumented 1464 ms update: 470 ms of kernel execution union, 544 ms including
copies, and only about 21 ms in grouped native FP4 GEMMs. Its 8292 forward
quantizer calls and 8292 total backward transform/requantization calls, plus
many small copies and host launches, identify preparation as the first target.
Profiler overhead can enlarge gaps; this trace is not an occupancy measurement
and is not substituted for unprofiled timing. Batching changes forward
quantization/packing, while the per-expert backward transform loop remains.

## Remaining gates and evidence

Keep this feature branch experimental. The next performance work should batch
backward Hadamard/EDEN transforms while retaining independent expert reduction
boundaries, and fuse scale/operand packing into their producers. Reuse the
decoded-operand and estimator checks before benchmarking each change. A matched
learning-recipe study and language-model heldout evaluation are still required
before claiming viable NVFP4 training or scaling to a larger model. New GEMMs
are not the first target supported by this trace.

Machine-readable results are in
[`results/native_nvfp4_backend.json`](results/native_nvfp4_backend.json).
[`results/native_nvfp4_backend_evidence.tar.gz`](results/native_nvfp4_backend_evidence.tar.gz)
retains 64 raw result, telemetry, log and source-hash files plus a SHA-256
manifest, verified after extraction from the archive. Failed setup and dtype
pilots are retained. The 325 MiB eager trace stays local at
`runs/performance/native-nvfp4/model-pilot.trace.json`; its size and SHA-256 are
recorded in the results. No model checkpoint was created or changed.

## Reproduction

Use only the repository `.venv`. Optional dependencies and their source fixes
are pinned in `NVFP4_FEASIBILITY.md` and `EXPERT_BACKENDS.md`; no dependency or
system upgrade is required. The CUDA include overlay supplies existing venv
cuRAND headers; it does not modify the system toolkit.

```sh
export PATH="$PWD/.venv/bin:$PATH"
export MAX_JOBS=4
export FLASHINFER_EXTRA_CUDAFLAGS="-I$PWD/runs/dependencies/cuda-curand-headers"
export QUACK_CACHE_DIR="$PWD/runs/dependencies/quack-expert-backends-test"

.venv/bin/python -m replication.gpu_monitor \
  --out runs/performance/native-nvfp4/suite.power.json -- \
  .venv/bin/python -m unittest discover -s tests -v

.venv/bin/python -m replication.qualify_nvfp4_backend --mode gradients \
  --out runs/performance/native-nvfp4/gradients-final.json
.venv/bin/python -m replication.qualify_nvfp4_backend --mode learning \
  --out runs/performance/native-nvfp4/learning-final.json
.venv/bin/python -m replication.qualify_nvfp4_backend --mode benchmark \
  --out runs/performance/native-nvfp4/benchmark-final.json
.venv/bin/python -m replication.qualify_nvfp4_backend --mode equivalence \
  --baseline c4bb969 \
  --out runs/performance/native-nvfp4/batching-equivalence-final.json
```

The qualification runner monitors its entire process and attaches power and
energy to explicit benchmark windows. Sensor averaging and sampled peaks limit
short-window precision. Board energy includes desktop GPU activity and excludes
CPU/system energy. Downloaded checkpoint files remain read-only in model runs.

# Native NVFP4 expert backend qualification

Work in progress on `feature/native-nvfp4-experts`, based on `dev` at
`33f5591c68958ae1a99ef5f513c20ab985db1e11`. The default backend remains
`reference`. This experiment is explicitly selected with
`MINAGI_EXPERT_BACKEND=nvfp4`; it is not yet a recommended training path.

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

Final measurements and retained evidence will be added after qualification.

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
  --out runs/performance/native-nvfp4/layer-final.json
```

The qualification runner monitors its entire process and attaches power and
energy to explicit benchmark windows. Sensor averaging and sampled peaks limit
short-window precision. Board energy includes desktop GPU activity and excludes
CPU/system energy. Downloaded checkpoint files remain read-only in model runs.

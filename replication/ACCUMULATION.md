# SM120 Tensor Core accumulation with Gefen

This research screen tests whether the RTX 5090's faster FP16-accumulating
Tensor Core path helps actual expert training. BF16/Sonic + Gefen remains the
measured production reference; no trainer defaults or checkpoint formats change.

## Existing kernels and the adaptation

SonicMoE uses QuACK GEMMs/fused epilogues plus Triton routing/reduction kernels.
The FP8 study also uses QuACK and PyTorch scaled GEMMs. This study reuses the
installed QuACK SM120 scheduling, shared-memory pipeline, MMA atoms, epilogue,
and split-K reduction. There is no new hand-written GEMM mainloop.

QuACK's compile helper passes `Float32` to its kernel constructor. Its shared
fragment allocation helper also allocates FP32 registers. `accum_experts.py`
changes those two choices process-locally for FP16 operands, using the existing
FP16 MMA atom and matching FP16 accumulator fragments. No installed package is
edited. Compile and dispatch plans are isolated from the FP32 path; the custom
compiler has a separate cache identity. The adapter is for isolated serial
research processes, with research kernels initialized before other GEMM JITs;
it is not a thread-safe production backend.

PyTorch's `allow_fp16_accumulation` flag controls its cuBLAS path; it does not
override QuACK's explicitly selected MMA accumulator. Changing that flag alone
would not test the accumulator choice in the Sonic/QuACK expert kernels.

The existing `SplitKMode.SEPARATE` path provides the two-stage candidate:
shorter FP16 partial sums are stored in FP32, then combined by the existing
FP32 reduction. This is four scheduled K splits, not a promise of an exact
fixed chunk length for every shape. Workspace/reduction costs are measured.

`inspect_accum_kernels.py` extracts the embedded CUDA ELF from QuACK's AOT
objects and uses the venv's existing `nvdisasm`. Disassembly distinguishes
`HMMA.16816.F16`, `HMMA.16816.F32`, and the BF16 operand variant. Output tensor
dtype alone is not evidence of accumulator instruction selection.

NVIDIA lists 209.5 dense TFLOPS for BF16/FP32 accumulation and 419 for
FP16/FP16 accumulation on RTX 5090. These are theoretical rates, not expected
whole-model speedups. The restriction is instruction-specific: native NVFP4
also accumulates in FP32 and has a different throughput ceiling.
[Architecture whitepaper, Table 3](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf#page=47).
Standard BF16 MMA uses FP32 accumulation; changing a stored `.grad` tensor's
dtype does not change that instruction.
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

## Compared recipes

| Research recipe | Matrix operands | MMA accumulation | K splits |
| --- | --- | --- | ---: |
| `bf16-fp32` | BF16 | FP32 | 1 |
| `fp16-fp32` | FP16 | FP32 | 1 |
| `fp16-fp16` | FP16 | FP16 | 1 |
| `fp16-split4` | FP16 | FP16 partials, FP32 final reduction | 4 |

The FP16 pair isolates accumulator precision while keeping operand casts and
other arithmetic fixed. All research recipes share padded expert dispatch.
Sonic BF16 provides the fused sparse production reference. Comparing a padded
recipe with Sonic measures the full implementation difference, not just MMA.

Master weights, outer accumulated weight gradients and Gefen's interface remain
FP32. Outputs and SwiGLU remain BF16; other model components retain the existing
mixed precision policy. This is an expert-only FP16 experiment, not global FP16
autocast or an FP16-master optimizer. There is no new loss-scaling recipe.
FP16 operand underflow/range limits and FP16 summation error are distinct risks.

Immutable cast-weight caches use tensor identity, storage, version, optimizer
and paging generations. Routing, checkpoint binding and replay ownership reuse
the qualified FP8 study's machinery, with its quantization and FP8 linear
functions replaced by casts and controlled 16-bit GEMMs. No FP8 arithmetic runs
inside these research recipes.

## Qualification and measurements

Eight focused tests cover cache ownership/invalidation, decoded-operand FP64
oracles for native forward/dgrad/wgrad (2D and batched), all mixture derivatives,
checkpoint replay across unversioned paging writes, inference-created caches,
causal prefixes, full-model cached/uncached replay equivalence, deterministic
split reduction, and an explicit FP16 overflow stress case.

A separate eight-update gate repeats all five backends in independent processes,
checking exact initial weights, first gradients, final weights, optimizer and
host-expert history, slots, objectives and held-out evaluations. Split-K also
replays without caching. These gates exercise actual Gefen updates and paging.

The learning protocol matches [the FP8 study](FP8_GEFEN.md): read-only original
undertrained checkpoint, hidden 512 / expert width 2048, 32 resident experts,
depth 6, context 512, 256 updates on each data-order seed 42/43/44, identical
starting weights and training windows. Gefen expert LR `3e-5`, AdamW auxiliary
LR `1e-4`, trunk LR `2e-5`, betas `(0.9,0.95)`, weight decay 0.1, gradient clip 1.
All touched experts remain cached in host RAM. No learned checkpoint is saved.
Before/after SHA256 hashes cover all 401 original checkpoint files.

TinyStories validation text supplies disjoint token ranges (adaptation before
262144, tuning from 262144, test from 524288; sixteen 512-token held-out windows).
These are byte/marker tokens, not canonical document-deduplicated splits;
pretraining overlap is unknown. Each backend evaluates in its own arithmetic.
Fixed rates and short data-order sweeps do not establish convergence or
precision-specific optimal hyperparameters.

Synthetic raw GEMMs compare decoded operands with FP64 results. Linear probes
include all three forward/backward GEMMs and casts. Expert probes include the
entire routed forward/backward, with cached unchanged weights. Current expert
geometry is 4096 input tokens, 32 experts, top-8, hidden 512 / width 2048. Large
geometry is 2048 tokens, four experts, top-2, hidden 4096 / width 16384. Balanced
synthetic routes do not model the padding imbalance of real routing. Synthetic
memory peaks are fixture measurements, not a full-model fit demonstration.

GPU processes run serially under external read-only NVML sampling, including
JIT, warmup, evaluation and tests. Timings use three seconds after warmup and
1.1 seconds of workload settling, repeated with reversed method order. Reports
include full-window rows/s, iteration medians, average/peak sampled watts,
board energy per input row, and peak allocated VRAM. Model measurements report
new language-model tokens/s and J/new token, over both all updates and the
predeclared interval after 16 updates. Sensor averaging, sampled peaks and
desktop activity limit power precision; CPU/system energy is excluded.

## Commands

Use the existing repository venv; no installation is required in the qualified
local environment. All commands below run from the repository root.

```sh
PATH="$PWD/.venv/bin:$PATH" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
MINAGI_ACCUM_TEST_CUDA=1 MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH=1 \
TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2 \
.venv/bin/python -m replication.gpu_monitor \
  --out runs/accum-tests.power.json -- \
  .venv/bin/python -m unittest tests.test_accum_experts -v
```

```sh
.venv/bin/python -m replication.run_accumulation_screen --phase probes
.venv/bin/python -m replication.run_accumulation_screen --phase repro
.venv/bin/python -m replication.run_accumulation_screen --phase learning
.venv/bin/python -m replication.summarize_accumulation
.venv/bin/python -m replication.inspect_accum_kernels \
  --out runs/accum-instructions.json
```

The runner supplies external monitoring and the qualified local Gefen build
opt-in. Cached learning results require matching configurations/source hashes;
timing probes require fresh outputs. Use `--root` consistently for a separate
study directory. Kernel inspection is CPU-only and executes no CUDA code.

## Results

The complete controlled measurements and final verification are pending.

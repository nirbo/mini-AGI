# SM120 Tensor Core accumulation with Gefen

This research screen tests whether the RTX 5090's faster FP16-accumulating
Tensor Core path helps actual expert training. BF16/Sonic + Gefen remains the
measured production reference; no trainer defaults or checkpoint formats change.

## Existing kernels and the adaptation

[SonicMoE](https://github.com/Dao-AILab/sonic-moe) uses
[QuACK](https://github.com/Dao-AILab/quack) GEMMs/fused epilogues plus Triton
routing/reduction kernels. QuACK builds on
[CUTLASS/CuTe DSL](https://github.com/NVIDIA/cutlass).
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
It does not test register-resident two-stage accumulation inside one CTA;
the cost of this global-workspace split-K method cannot rule out that design.

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
Seed-42 long runs check their first eight objectives/gradient norms against
the gate. The complete 256-update trajectories are not independently replayed.

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

Sonic BF16 remains the best measured choice for the current checkpoint: **3643
new tokens/s**, **0.604345 mean test CE**, and **0.0499 J/new token** after the
first 16 updates. The padded FP16-accumulating prototype reaches **3529 tokens/s**,
**0.630593 CE**, and **0.0542 J/new token**. With otherwise matched FP16 operands,
FP16 accumulation increases final test CE on all three seeds versus FP32
accumulation (mean **0.609397**). This is evidence against switching to this
recipe now, not a convergence claim about every FP16 training recipe.

The hardware opportunity is real: the large synthetic expert layer takes
**20.77–20.80 ms** with FP16 accumulation, versus **28.81–28.86 ms** for the
matched FP16/FP32 path and **32.19–32.31 ms** for Sonic BF16. That is **1.39×**
and **1.55×** faster respectively, while retaining FP32 master gradients. At the
current layer size Sonic takes **5.02–5.03 ms**, versus **7.19–7.22 ms** for the
padded FP16 prototype. Uniform four-way split-K is slower at both sizes.

Further development should expose accumulator selection in the existing fused
kernel path, then test selective policies (for example, retaining FP32 weight-
gradient GEMMs) before a production integration. Register-resident two-stage
accumulation is another untested candidate; the global-workspace split-K result
does not settle its performance. The next production optimization still needs
an end-to-end trace to identify the current model's dominant costs.

These measurements distinguish the internal Tensor Core accumulation rate from persistent FP32 gradient storage. All candidates retain FP32 masters and outer gradients. The research backend remains isolated from production trainer choices; no default is changed by this study.

### Matched learning screen

| Seed | Sonic BF16 | Padded BF16/FP32 | FP16/FP32 | FP16/FP16 | FP16 split-K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.574349 | 0.579617 | 0.579539 | 0.607105 | 0.644041 |
| 43 | 0.648714 | 0.663965 | 0.638583 | 0.656855 | 0.649449 |
| 44 | 0.589972 | 0.594154 | 0.610070 | 0.627821 | 0.604119 |
| Mean | 0.604345 | 0.612579 | 0.609397 | 0.630593 | 0.632536 |

Values are held-out test CE (lower is better), evaluated in each backend arithmetic. These short fixed-rate trials do not establish long-run convergence or optimal rates; no common-arithmetic learned-weight cross-evaluation was performed.

| Interval | Backend | Seconds | New tokens/s | Avg / peak W | J/new token | Peak GiB¹ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| All 256 updates | `sonic` | 46.63 | 2811 | 175 / 194 | 0.0621 | 1.64 |
| All 256 updates | `bf16-fp32` | 47.73 | 2746 | 184 / 215 | 0.0669 | 1.74 |
| All 256 updates | `fp16-fp32` | 48.48 | 2704 | 186 / 217 | 0.0690 | 1.77 |
| All 256 updates | `fp16-fp16` | 47.83 | 2741 | 183 / 207 | 0.0667 | 1.77 |
| All 256 updates | `fp16-split4` | 59.30 | 2210 | 215 / 259 | 0.0974 | 2.27 |
| Updates 17–256 | `sonic` | 33.74 | 3643 | 182 / 194 | 0.0499 | 1.64 |
| Updates 17–256 | `bf16-fp32` | 34.66 | 3545 | 193 / 215 | 0.0545 | 1.74 |
| Updates 17–256 | `fp16-fp32` | 35.38 | 3474 | 196 / 217 | 0.0564 | 1.77 |
| Updates 17–256 | `fp16-fp16` | 34.82 | 3529 | 191 / 207 | 0.0542 | 1.77 |
| Updates 17–256 | `fp16-split4` | 45.61 | 2694 | 228 / 259 | 0.0847 | 2.27 |

Means are over three seeds; peaks are maxima. ¹Memory peaks cover the complete training interval and are repeated in both rows. Training intervals exclude loading/evaluation. The after-16 interval can still include paging and late initialization. Whole-process power samples include all phases.

### Synthetic forward/backward performance

Timing ranges span two repeats with reversed method order. Rows/s and power/energy are means over full measurement windows; peak power and memory are maxima. Row throughput is not the reciprocal of median iteration time. Synthetic cached weights stay unchanged; there is no optimizer in these loops.

| Raw GEMM M×K×N | Recipe | Time ms | Input rows/s | Avg / peak W | mJ/input row | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096×4096×4096 | `bf16-fp32` | 0.673–0.673 | 6075760 | 574 / 586 | 0.094 | 0.38 |
| 4096×4096×4096 | `fp16-fp32` | 0.677–0.678 | 6038834 | 598 / 602 | 0.099 | 0.38 |
| 4096×4096×4096 | `fp16-fp16` | 0.458–0.458 | 8920514 | 563 / 576 | 0.063 | 0.38 |
| 4096×4096×4096 | `fp16-split4` | 0.763–0.765 | 5318367 | 552 / 559 | 0.104 | 0.62 |

Raw GEMM outputs are FP32 for all recipes. Errors against FP64 multiplication of the same decoded operands isolate accumulation error from operand casting:

| Raw GEMM M×K×N | Recipe | Relative L2 error range |
| --- | --- | ---: |
| 128×512×2048 | `bf16-fp32` | 4.73e-07–4.73e-07 |
| 128×512×2048 | `fp16-fp32` | 6.47e-07–6.47e-07 |
| 128×512×2048 | `fp16-fp16` | 0.000847–0.000847 |
| 128×512×2048 | `fp16-split4` | 0.00044–0.00044 |
| 128×2048×512 | `bf16-fp32` | 2.1e-06–2.1e-06 |
| 128×2048×512 | `fp16-fp32` | 2.45e-06–2.45e-06 |
| 128×2048×512 | `fp16-fp16` | 0.00166–0.00166 |
| 128×2048×512 | `fp16-split4` | 0.000843–0.000843 |
| 2048×128×512 | `bf16-fp32` | 1.2e-07–1.2e-07 |
| 2048×128×512 | `fp16-fp32` | 1.89e-07–1.89e-07 |
| 2048×128×512 | `fp16-fp16` | 0.00044–0.00044 |
| 2048×128×512 | `fp16-split4` | 0.000328–0.000328 |
| 4096×4096×4096 | `bf16-fp32` | 4.4e-06–4.4e-06 |
| 4096×4096×4096 | `fp16-fp32` | 4.86e-06–4.86e-06 |
| 4096×4096×4096 | `fp16-fp16` | 0.00235–0.00235 |
| 4096×4096×4096 | `fp16-split4` | 0.00118–0.00118 |

| Linear forward/backward M×K×N | Recipe | Time ms | Input rows/s | Avg / peak W | mJ/input row | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128×512×2048 | `bf16-fp32` | 0.101–0.164 | 1015020 | 135 / 141 | 0.139 | 0.08 |
| 128×512×2048 | `fp16-fp32` | 0.107–0.107 | 1065880 | 139 / 142 | 0.131 | 0.08 |
| 128×512×2048 | `fp16-fp16` | 0.082–0.169 | 1078486 | 134 / 145 | 0.126 | 0.08 |
| 128×512×2048 | `fp16-split4` | 0.107–0.251 | 875975 | 152 / 168 | 0.189 | 0.10 |
| 128×2048×512 | `bf16-fp32` | 0.094–0.156 | 1136842 | 134 / 141 | 0.121 | 0.08 |
| 128×2048×512 | `fp16-fp32` | 0.097–0.098 | 1205893 | 138 / 143 | 0.115 | 0.08 |
| 128×2048×512 | `fp16-fp16` | 0.069–0.162 | 1352802 | 135 / 149 | 0.103 | 0.08 |
| 128×2048×512 | `fp16-split4` | 0.108–0.108 | 972692 | 147 / 161 | 0.153 | 0.10 |
| 4096×4096×4096 | `bf16-fp32` | 2.230–2.231 | 1833550 | 568 / 576 | 0.310 | 0.66 |
| 4096×4096×4096 | `fp16-fp32` | 2.286–2.287 | 1788211 | 591 / 593 | 0.330 | 0.69 |
| 4096×4096×4096 | `fp16-fp16` | 1.537–1.537 | 2659509 | 600 / 605 | 0.226 | 0.69 |
| 4096×4096×4096 | `fp16-split4` | 2.540–2.549 | 1604029 | 547 / 565 | 0.341 | 0.94 |

| Expert hidden / width | Backend | Time ms | Input tokens/s | Avg / peak W | mJ/input token | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 2048 | `sonic` | 5.018–5.030 | 814327 | 567 / 574 | 0.696 | 1.41 |
| 512 / 2048 | `bf16-fp32` | 7.242–7.256 | 564398 | 525 / 532 | 0.930 | 1.49 |
| 512 / 2048 | `fp16-fp32` | 8.150–8.188 | 500770 | 526 / 531 | 1.051 | 1.53 |
| 512 / 2048 | `fp16-fp16` | 7.190–7.216 | 567982 | 534 / 540 | 0.940 | 1.53 |
| 512 / 2048 | `fp16-split4` | 18.066–18.077 | 226231 | 441 / 446 | 1.948 | 2.34 |
| 4096 / 16384 | `sonic` | 32.194–32.312 | 63438 | 596 / 600 | 9.401 | 8.63 |
| 4096 / 16384 | `bf16-fp32` | 27.765–27.906 | 73435 | 598 / 600 | 8.148 | 7.94 |
| 4096 / 16384 | `fp16-fp32` | 28.808–28.855 | 70885 | 600 / 601 | 8.465 | 8.06 |
| 4096 / 16384 | `fp16-fp16` | 20.770–20.795 | 98461 | 599 / 600 | 6.085 | 8.06 |
| 4096 / 16384 | `fp16-split4` | 49.110–49.149 | 41643 | 506 / 511 | 12.154 | 12.06 |

The large fixture has 805,306,368 expert parameters. Its memory peak covers this layer, casts, FP32 gradients and activations, not a complete recurrent model or optimizer state. Balanced routing and repeated unchanged weights can differ substantially from real-model work.

### Numerical limits and kernel-development implications

Native FP16 accumulation is an actual hardware change, confirmed by SASS, not a storage-dtype flag. The stress test deliberately constructs a finite final sum whose full FP16 partial sums overflow; split-K avoids overflow in that case. Passing normal-input/short-learning checks does not eliminate this range risk. The whole expert path also casts operands to FP16 without a new loss-scaling policy.

The existing-kernel adaptation is small: expose the accumulator type and allocate matching register fragments. Further work should retain the upstream SM120 pipeline and first target measured shape-specific scheduling, casts/packing and fused epilogues/routing. Uniform split-K pays extra workspace and reduction work; a win on one isolated GEMM does not justify applying it to every forward/backward projection. A production Sonic FP16 path would require its own fused forward/backward and numerical qualification; this padded adapter is not that implementation.

### Verification and evidence

The focused **eight-test qualification** (six CUDA, two CPU) and the full **127-test regression suite** passed without skips. Eleven replay/cache runs, fifteen matched 256-update trials and 76 synthetic timing cases completed. All 401 original checkpoint files are unchanged. Installed QuACK/Sonic/Gefen source hashes match the preceding FP8 study; no packages or system configuration were changed.

Machine-readable [results](results/accumulation.json), [verification/environment](results/accumulation_verification.json), and [raw evidence archive](results/accumulation_evidence.tar.gz) include timestamped NVML samples, per-step traces, source snapshots, AOT cubins/SASS and labeled excluded pilots. Sensor averaging and desktop activity affect power; CPU energy is excluded.

# Complete-layer NVFP4 crossover on RTX 5090

The current experimental backend has a measured competitive region against
Sonic BF16: hidden 4096, expert width 16384, four resident experts/top-4 and
4096–8192 token rows per expert. This fits the 24 GiB test budget with full
FP32 AdamW state. It does not establish full recurrent-model fit or quality.
The feature branch remains experimental and unmerged.

## Workload and correctness

Source `533edb0`, `replication/sweep_nvfp4_crossover.py`. Each process constructs
synthetic FP32 weight slabs, FP32 gradients and both AdamW moments, plus a BF16
compute cache with separate per-use gradient edges. Timings include the entire
three-projection SwiGLU layer forward/backward, routing-score/input gradients,
fused AdamW and compute-cache refresh. Fixed balanced routes isolate arithmetic.
Repeated inputs and a synthetic output gradient are a performance workload,
not a matched learning experiment. BF16/FP4 initialization is identical; timing
loops can perform different numbers of updates.

Every requested FP4 projection shape is first checked at one expert against
independently decoded FP4 operands, including forward, input-gradient and
weight-gradient GEMMs in both projection directions. Acceptance is finite
results and relative L2 error ≤0.005. Existing tests cover grouping, unequal
expert scales, unused experts and causality. The two new CPU workload tests
independently check assignment counts, unique experts and token order.

Each configuration uses Sonic/FP4/FP4/Sonic ordering, three warmup iterations,
1.1 seconds of workload settling and a three-second measurement window.
The allocator cap is at most 24 GiB, 85% of total VRAM or initially free VRAM
minus 4 GiB, whichever is smallest. It is local to the process. The budget
leaves desktop/additional-model headroom and is not the card's physical limit.
No system, GPU clock or power-limit settings change.

## Measurements

Times are ranges of per-run medians, including optimizer/cache refresh. Power
and energy are aggregated over the two measured windows for each backend.
`M` is accepted token rows **per expert**, not total input tokens. Top-k is 8
except the four-expert configurations, where it is 4.

| Hidden / FF width | Experts | M | Sonic ms | FP4 ms | FP4 throughput / Sonic | FP4 peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 2048 | 32 | 1024 | 6.26–6.27 | 25.06 | 0.250× | 2.77 |
| 512 / 2048 | 32 | 4096 | 14.89 | 55.48–55.49 | 0.268× | 6.07 |
| 1024 / 4096 | 32 | 1024 | 23.68–23.69 | 47.34–47.43 | 0.500× | 8.47 |
| 1024 / 4096 | 32 | 4096 | 56.09–56.13 | 119.51–119.54 | 0.469× | 15.05 |
| 1536 / 6144 | 32 | 1024 | 52.58–52.62 | 82.24–82.27 | 0.640× | 17.25 |
| 1536 / 6144 | 32 | 4096 | 124.57–124.62 | exceeds allocation cap | — | — |
| 1536 / 6144 | 16 | 4096 | 62.56–62.61 | 98.44–98.45 | 0.636× | 13.63 |
| 2048 / 8192 | 16 | 1024 | 46.59–46.63 | 64.29 | 0.725× | 14.61 |
| 2048 / 8192 | 16 | 4096 | 110.11–110.15 | 144.15–144.20 | 0.764× | 21.19 |
| 4096 / 4096 | 8 | 4096 | 55.07–55.08 | 67.88–67.89 | 0.812× | 9.69 |
| 4096 / 8192 | 4 | 4096 | 54.94–55.00 | 55.37–55.40 | 0.993× | 8.74 |
| 4096 / 16384 | 4 | 4096 | 109.33–109.41 | 97.84 | **1.118×** | **16.87** |
| 4096 / 16384 | 4 | 8192 | 195.71–195.75 | 159.80–159.88 | **1.224×** | **21.31** |

The 2048/8192/32-expert bank is skipped before allocating: masters, gradients,
moments and BF16 cache alone require 27 GiB, above the test budget. The
1536/6144/32-expert large-M FP4 failures occur during full-layer training after
their independent projection checks pass. They are allocation-cap failures,
not evidence that the same layer cannot run using more of the physical card.

At 4096/8192 with four experts FP4 is near throughput parity and costs about
21.5% less board energy per input row. At 4096/16384 it is faster and more
efficient:

| M | Backend | Average board W | Board energy / input row, relative to Sonic | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: |
| 4096 | Sonic | 600 | 1.000× | 14.76 |
| 4096 | FP4 | 474 | **0.708×** | 16.87 |
| 8192 | Sonic | 600 | 1.000× | 17.52 |
| 8192 | FP4 | 485 | **0.660×** | 21.31 |

Absolute energy/input-row, sampled peaks, utilization, clocks, temperature,
limiting masks and reserved memory are retained in the machine-readable data.
Sensor averaging and short windows limit power precision. Peak allocated memory
is reset after warmup/settling; optimizer state and cache are already allocated
and remain included. CUDA libraries/driver and desktop use can add memory beyond
PyTorch's allocator statistics. No GPU OOM is deliberately launched above the
budget, and no zero-K native GEMM is launched.

## Interpretation

For this implementation, 4096/8192/four-expert geometry is near parity, and
4096/16384 at M≥4096 wins among the sampled points. This is a **sampled
crossover region**, not an exact boundary or a universal parameter threshold.
Fused preparation could move it toward smaller matrices.

The 2048/8192/16-expert and 4096/16384/four-expert banks both contain about
805 million resident expert parameters and have the same nominal matrix
multiply work at fixed M. Sonic takes about 110 ms in both cases; FP4 falls
from about 144 ms to 98 ms with fewer, wider experts. Matrix geometry and
preparation costs explain why parameter count alone is insufficient.

The winning bank's state/cache floor is 13.5 GiB. The measured layer/optimizer
peaks leave room for additional work on a 32 GiB card, but attention, prelude,
recurrent unrolling, model activations and checkpointing remain unmeasured at
this width. Fewer experts/top-k also change the architecture and admission
behavior. The existing downloaded checkpoint cannot simply be widened or
treated as quality-equivalent to this synthetic geometry.

## Reproduction and evidence

Use the repository `.venv` and the same optional dependencies/patches as
`NATIVE_NVFP4_BACKEND.md`. Example single process (internally power monitored):

```sh
PATH="$PWD/.venv/bin:$PATH" MAX_JOBS=4 \
FLASHINFER_EXTRA_CUDAFLAGS="-I$PWD/runs/dependencies/cuda-curand-headers" \
QUACK_CACHE_DIR="$PWD/runs/dependencies/quack-expert-backends-test" \
.venv/bin/python -m replication.sweep_nvfp4_crossover \
  --hidden 4096 --width 16384 --experts 4 --topk 4 \
  --rows-per-expert 4096 --backend nvfp4 \
  --out runs/performance/nvfp4-crossover/reproduction.json
```

Raw measurements and summaries:
[`results/nvfp4_crossover.json`](results/nvfp4_crossover.json),
[`results/nvfp4_crossover_evidence.tar.gz`](results/nvfp4_crossover_evidence.tar.gz).
The archive includes raw results, timestamped power samples, logs and a verified
SHA-256 manifest. Fourteen configurations produce 53 primary processes, plus a
retained earlier pilot. Model checkpoints are neither read nor written.

# RTX 5090 native NVFP4 feasibility experiment

Native W4A4 forward **and both backward GEMMs work on this RTX 5090**.
Keep NVFP4 as the target, with a replaceable expert-compute backend and the
existing BF16 implementation as the numerical reference. The stock Quartet-II
linear path is not yet a suitable drop-in optimization for the current small
experts. This experiment does not enable a new production training backend.

## Scope and acceptance

The probes use synthetic tensors at the checkpoint's hidden width 512 and
expert width 2048, plus a larger 4096-square scaling case. FP32 master weights
and accumulated gradients are retained. Quartet's forward, input-gradient and
weight-gradient matrix operands are genuinely NVFP4; backward quantization is
enabled. This is neither fake quantization nor frozen-base adapter training.
No checkpoint was written or changed.

`probe_nvfp4.py` independently decodes E2M1 nibbles, E4M3 block scales and the
128x4 physical scale layout into FP32. Each native GEMM must be finite and have
relative L2 error at most 0.005 against multiplication of **those same decoded
operands**. Lossless constant operands must match exactly. Error against the
original BF16 operands is reported separately. CPU tests cover nibble order,
sign, global scaling and scale indices across row/column tile boundaries.
This separates kernel arithmetic correctness from lossy quantization.

For Quartet backward, the probe intercepts all three actual FP4 GEMMs and
checks each against the independent decoder. It also reports mean gradients
over 32 stochastic backward passes against BF16 and against the decoded
forward's straight-through gradient reference. These differences describe an
estimator; they do not prove unbiasedness or long-run learning quality.

`probe_moe_backends.py` compares a full BF16 SwiGLU expert layer's forward and
backward, including input, three weight and routing-score gradients, with a
per-expert reference. Routing stays external and fixed. The declared tolerance
is finite output/gradients with relative L2 error at most 0.025, reflecting
different BF16 reduction/fusion rounding. A separate case exercises 16 unused
experts and checks their gradients are zero. This is a synthetic layer check,
not a substitute for model causality, paging and learning tests.

## Measured results

**Complete linear forward + backward** (M × K × N):

| Shape | Method | ms | Rows/s | Average / peak W | µJ/row | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128 × 512 × 2048 | bf16 | 0.056–0.057 | 1584126–1950461 | 190–202 / 209–211 | 103.7–119.6 | 0.14 |
| 128 × 512 × 2048 | native_nvfp4 | 0.912–1.125 | 116087–132049 | 104–107 / 111–120 | 813.3–898.8 | 0.14 |
| 1024 × 512 × 2048 | bf16 | 0.068–0.108 | 10194122–14751728 | 270–382 / 298–391 | 25.9–26.5 | 0.16 |
| 1024 × 512 × 2048 | native_nvfp4 | 0.910–1.056 | 999036–1090116 | 111–113 / 192–198 | 101.9–112.7 | 0.16 |
| 128 × 2048 × 512 | bf16 | 0.056–0.114 | 1151784–1899127 | 161–205 / 167–220 | 108.2–140.0 | 0.14 |
| 128 × 2048 × 512 | native_nvfp4 | 0.921–1.116 | 121685–130048 | 106 / 110–119 | 815.0–873.3 | 0.14 |
| 1024 × 2048 × 512 | bf16 | 0.071–0.072 | 11843296–14215893 | 387–404 / 411–414 | 28.4–32.7 | 0.19 |
| 1024 × 2048 × 512 | native_nvfp4 | 1.117–1.132 | 930885–974494 | 113–116 / 150–195 | 116.1–124.7 | 0.19 |
| 4096 × 4096 × 4096 | bf16 | 2.145–2.147 | 1906054–1907414 | 592–596 / 600–604 | 310.7–312.5 | 1.17 |
| 4096 × 4096 × 4096 | native_nvfp4 | 1.114–1.287 | 3272842–3449711 | 395–416 / 441–467 | 120.6 | 1.19 |

**Complete BF16 expert layer** (4096 original tokens, 32 experts, top-8, hidden 512, width 2048):

| Workload | Method | ms | Input tokens/s | Average / peak W | µJ/input token | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Expert layer | reference | 8.139–8.161 | 501248–502783 | 499–507 / 504–508 | 993.4–1012.4 | 1.83 |
| Expert layer | native_grouped | 7.924–7.925 | 516582–516931 | 501–509 / 502–536 | 969.7–984.6 | 1.93 |
| Expert layer | sonic | 4.605–4.615 | 886525–888465 | 596–598 / 600 | 672.4–672.9 | 1.75 |

**Packed grouped FP4 forward only** (32768 logical assignments):

| Workload | Method | ms | Assignment rows/s | Average / peak W | µJ/assignment | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 35072 padded rows | cudnn | 0.165 | 197359748 | 542 / 553 | 2.7 | 1.21 |
| 32808 padded rows | cutlass | 0.142 | 229317040 | 582 / 584 | 2.5 | 1.16 |

The unchanged real-checkpoint BF16 model baseline (8 warmups, 32 measured updates, fresh optimizer, context 4096, stride 2048, fixed depth 24, expert checkpointing and weight cache enabled) measured **0.377 s/update**, **5,439 new tokens/s** from median step time and **5.16 GiB** peak allocation. Its aligned measurement window averaged **268 W**, sampled peak **304 W**, and **0.0500 J/new token**. This is a baseline only; no FP4/Sonic model-speedup claim follows.

Both native and reference probes passed: **69 repository tests and 70 selected probe cases**. Typical native GEMM relative L2 error against decoded operands was about 0.00166; original-operand error also includes quantization (about 0.123 for the random dense forward cases). The causal diagnostic produced prefix-output relative L2 change **0.08748** with dynamic whole-matrix scaling and **0** with the fixed bound.

Synthetic held-out MSE after 400 updates: BF16 **1.32654e-05**; BF16 training followed by the same NVFP4 inference/PTQ path **0.0151477**; native NVFP4 training/inference **0.0241838**.

Exact per-run values and provenance: [`results/nvfp4_rtx5090_feasibility.json`](results/nvfp4_rtx5090_feasibility.json). Raw result JSON, power samples, suite log and SHA256 manifest: [`results/nvfp4_rtx5090_evidence.tar.gz`](results/nvfp4_rtx5090_evidence.tar.gz).

All timings are synchronized wall time, including Python/launch overhead.
Complete linear training includes conversion, quantization and backward;
complete expert-layer timing additionally includes dispatch and weight packing.
Neither includes an optimizer update. Packed FP4 GEMM timing excludes packing
and quantization and cannot be compared directly with complete BF16 training.
The BF16 layer baseline converts weights per forward; the production BF16
weight cache could change the comparison.

Each measured phase follows warmup and at least 1.1 seconds running the same
workload to reduce contamination from the sensor's previous averaging window.
Three-second windows are repeated twice for complete linear/expert training;
linear shape order and MoE backend order are reversed in the second run.
These observations establish a substantial size-dependent difference, not a
precise hardware ceiling or an end-to-end model speedup.

The synthetic tables' throughput and joules per row refer to linear input rows
or original MoE-layer input tokens; grouped forward uses expanded expert
assignment rows. Only the separate model baseline measures full language-model
new-token throughput. VRAM peaks include live reference
tensors and probe buffers and are not production memory requirements.

NVML board power is sampled every 100 ms and integrated over the explicit
measurement interval, with interpolated boundaries. It can be internally
averaged; sampled peaks are not instantaneous maxima. Board energy includes
desktop GPU activity and excludes CPU/system energy. Raw samples also record
utilization, clocks, temperature and clock-limiting reason masks. No clocks or
power limits were changed. Low watts alone do not establish unused compute.

## Integration blockers demonstrated

**Default whole-matrix activation scaling violates causal prefix invariance.**
Appending high-amplitude future rows changes earlier quantized activations and
outputs. The diagnostic compares a 128-row prefix alone with that same prefix
followed by 128 rows of value 100. A fixed input bound makes prefix decoding
and outputs identical in this case. The bound of 128 in this diagnostic is a
proof of the mechanism, not a proposed model calibration. A real integration
needs fixed/calibrated or prefix-safe activation scales, plus clipping/drift
checks across recurrent depths and train/eval paths. Backward reductions over
training examples are a different issue from forward causality.

**Grouped scale layouts are backend-specific.** CUTLASS requires data offsets
divisible by four and scale-row offsets
`floor((data_offset + expert_index * 127) / 128) * 128`.
The cuDNN path is qualified here using 128-row data boundaries and global
128x4 scales, following the subset covered by its upstream tests. For logical
counts `[1,31,127,225]`, CUTLASS executes 392 rows while this cuDNN adapter
executes 640. Both pass with their own layouts. Early attempts sharing one
layout between them failed; those failures do not demonstrate a vendor kernel
bug. The 32-expert case includes an empty expert and uneven counts.
These are grouped **forward** tests; grouped FP4 dgrad/wgrad with variable
reduction sizes have not been implemented or qualified. No zero-K grouped
backward kernel was launched.

The grouped probe uses shared tensor-level weight/input scales and equal
`alpha` values across experts. Unequal per-expert global scales still need a
separate check. A model adapter must also keep weight scales stable for a
prefix when the resident expert bank changes; recomputing a shared maximum
over newly admitted experts could otherwise introduce another dependency on
future routing. This is an integration constraint, not a tested model fix.

**The toy learning result does not establish a QAT advantage over PTQ.**
The controlled regression uses the same initial FP32 weights, batch IDs, 400
AdamW updates (constant LR 0.003), 3072 training examples and 1024 held-out
examples. Native Quartet quantizes forward and backward; activation bound 8
is fixed. Both paths must reduce held-out compute MSE by at least 75%.
The BF16-trained weights are then evaluated through exactly the same NVFP4
inference path as the NVFP4-trained weights. The measured comparison above
does not favor NVFP4 training. It is one seed, one linear regression and one
untuned schedule, not an LLM convergence result or a general rejection of QAT.

## Backend direction

1. Introduce an expert-compute interface while retaining existing routing,
   admission, paging, optimizer ownership and BF16 reference behavior. Keep
   FP32 masters and optimizer state initially; this experiment does not test
   BF16 master updates.
2. Use SonicMoE/QuACK as a measured BF16 implementation candidate, without
   making its API the model's permanent precision contract. Before enabling
   it, verify checkpoint recomputation, cached-weight gradient edges, page
   replacement and causal model outputs on the real checkpoint.
3. Keep the NVFP4 candidate independent: combine qualified SM120 grouped
   forward primitives with a validated quantized-backward estimator and
   causal activation scaling. First measure a complete small expert layer;
   packing, quantization, transposes and launch costs must be included.
4. Run matched real-text BF16/native-FP4 training with held-out predictive CE,
   clipping statistics and quantized-inference evaluation. Only enable FP4
   after these pass and throughput/energy improve at the intended size.

This avoids treating a successful large dense GEMM as proof that the current
recurrent paged MoE is faster or trains equally well in FP4. It also preserves
the route toward native W4A4 as the model grows.

## Reproduction and dependencies

All installations were inside the repository `.venv`; no pre-existing package
version changed. Python 3.12.14, Torch 2.14.1+cu130, Triton 3.8.0, cuDNN
9.24.0.43, driver 595.91.07 and the **already installed** nvcc 13.1.115 were used.
The full package inventory and exact source revisions are in
[`results/nvfp4_environment.json`](results/nvfp4_environment.json). This is an
inventory, not a request to change system CUDA or drivers.

The probe-only dependencies are optional; ordinary model training does not
import them. Starting with the existing project venv, save constraints first:

```bash
uv pip freeze --python .venv/bin/python > runs/pre-nvfp4-constraints.txt
uv pip install --python .venv/bin/python -c runs/pre-nvfp4-constraints.txt \
  nvidia-ml-py==13.615.71 nvidia-cutlass-dsl==4.8.1
uv pip install --python .venv/bin/python -c runs/pre-nvfp4-constraints.txt \
  nvidia-cutlass-dsl-libs-cu13==4.8.1
uv pip install --python .venv/bin/python -c runs/pre-nvfp4-constraints.txt \
  flashinfer-python==0.7.2 scipy==1.17.1 nvtx==0.2.16 \
  pytest==9.1.1 parameterized==0.9.0 rich==15.0.0 \
  scikit-build-core==1.1.1 nanobind==3.1.0 cmake==4.4.4
```

Clone QuACK, SonicMoE and Quartet-II into a local dependency directory and
check out the exact commits in the inventory. Install QuACK and SonicMoE
from those source trees with `uv pip install --python .venv/bin/python
--no-deps <quack-path> <sonic-path>`. The source revisions were tested with
the existing Torch; the older published SonicMoE wheel's Torch pins are not
the tested configuration. Do not run upstream's system/conda setup scripts.

Apply [`patches/quartet2-inline-ptx-type.patch`](patches/quartet2-inline-ptx-type.patch)
with `git -C <quartet-path> apply <absolute-patch-path>`. It changes six
header variable definitions to `inline constexpr`, resolving duplicate symbol
link failures with this compiler. It changes no quantization arithmetic.
Build against the existing toolkit:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDAARCHS=120a \
  CMAKE_ARGS='-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc' \
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
  <quartet-path>/kernels
```

FlashInfer's CUTLASS JIT also needs cuRAND headers. On this desktop they are
already inside the venv's `nvidia-curand` package. Create a cuRAND-only include
overlay, avoiding unrelated runtime headers from a different CUDA release:

```bash
.venv/bin/python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
overlay = Path('runs/dependencies/cuda-curand-headers')
overlay.mkdir(parents=True, exist_ok=True)
dist = distribution('nvidia-curand')
for file in dist.files:
    if file.name.startswith('curand') and file.name.endswith('.h'):
        target = overlay / file.name
        if not target.exists():
            target.symlink_to(dist.locate_file(file).resolve())
PY
export MAX_JOBS=4
export FLASHINFER_EXTRA_CUDAFLAGS="-I$PWD/runs/dependencies/cuda-curand-headers"
```

The probes put this venv's build executables on their process-local PATH and
monitor all GPU work, including initial JIT compilation. Run GPU probes
sequentially, with no competing GPU jobs:

```bash
.venv/bin/python -m replication.probe_nvfp4 --mode kernel \
  --seconds 3 --out runs/performance/nvfp4/kernels.json
.venv/bin/python -m replication.probe_nvfp4 --mode grouped \
  --seconds 3 --out runs/performance/nvfp4/grouped.json
.venv/bin/python -m replication.probe_nvfp4 --mode training --seconds 3 \
  --shapes 128,512,2048 1024,512,2048 128,2048,512 1024,2048,512 4096,4096,4096 \
  --out runs/performance/nvfp4/training.json
.venv/bin/python -m replication.probe_nvfp4 --mode causality \
  --out runs/performance/nvfp4/causality.json
.venv/bin/python -m replication.probe_nvfp4 --mode learning --learn-steps 400 \
  --out runs/performance/nvfp4/learning.json
.venv/bin/python -m replication.probe_moe_backends --tokens 32 --empty-experts \
  --seconds 0 --out runs/performance/nvfp4/moe-empty.json
.venv/bin/python -m replication.probe_moe_backends --tokens 4096 --seconds 3 \
  --out runs/performance/nvfp4/moe.json
.venv/bin/python -m replication.gpu_monitor \
  --out runs/performance/nvfp4/suite.power.json -- \
  .venv/bin/python -m unittest discover -s tests -v
```

The scripts return nonzero on failed numerical checks. JSON contains exact
arguments, package versions, Git state, executed source hashes, timing windows
and error metrics; companion `.power.json` files preserve raw telemetry.
GPU/library errors are saved with tracebacks. The final evidence archive and
machine-readable summary accompany this report; earlier pilot failures remain
in `runs/performance/nvfp4/` locally and are excluded from performance claims.

Upstream references:

- [Quartet-II source and SM120 kernels](https://github.com/IST-DASLab/Quartet-II/tree/5f2a47e8f464b17ef4b1cac78db8d14edd1f3f7d)
- [SonicMoE tested source](https://github.com/Dao-AILab/sonic-moe/tree/7396f3e604827d8186c2e16e64b28ee33d3defd0)
- [QuACK tested source](https://github.com/Dao-AILab/quack/tree/77a750c0d9548d4a6e23c0f64062d61b3ea420ff)
- [CUTLASS grouped FP4 scale packing test](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/tests/gemm/test_group_gemm_fp4.py)
- [cuDNN grouped FP4 test coverage](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/tests/grouped_mm/test_grouped_mm_fp4.py)

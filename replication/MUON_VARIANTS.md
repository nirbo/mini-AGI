# Muon variant screen on RTX 5090

This is a research comparison, not a new trainer optimizer selection. The
trainer's default remains AdamW, with its existing optional Gefen backend.
The experiment uses FP32 masters/gradients and Sonic BF16 expert computation.
It does not qualify native NVFP4 convergence or performance.

## Candidates and implementation

- `adamw`: existing fused Torch AdamW with paged expert slabs.
- `gefen`: existing qualified PagedGefen adapter, Gefen 1.2.3 expert matrices
  and fused AdamW auxiliary parameters.
- `muon`: Microsoft Dion's batched Muon, Nesterov momentum 0.95, classic
  five-step Newton–Schulz, Adam-compatible shape scaling (`rms_norm`).
- `muon_plus`: the MuonPlus author's complete optimizer, Keller NS5,
  `norm_mode=col_row`, Nesterov 0.95, and its own `rms_scaling=True` shape
  scaling. This preserves the author's formula rather than substituting
  Microsoft's different scaling. Its implementation processes matrices
  individually; timing therefore compares available implementations, not
  algorithmic complexity alone.
- `normuon`: Microsoft Dion's batched NorMuon, Nesterov 0.95, per-neuron
  second-moment decay 0.95, classic NS5 and `rms_norm` scaling.
- `dion3_quarter` / `dion3_half`: Microsoft Dion3 (NorDion2), row-selection
  fraction 0.25 / 0.5, momentum 0.95, neuron decay 0.95, classic NS5 and
  `rms_norm` scaling. Dion3's momentum/error feedback rule differs from
  Nesterov Muon; this is part of the algorithm being tested.

Classic NS5 uses coefficients `(3.4445, -4.7750, 2.0315)` at all five
iterations, preserving the author's BF16 operation ordering. The current
Microsoft function named `zeropower_via_newtonschulz5` uses a different
sequence; we supply a custom function explicitly. No Gram NS dependencies,
custom GEMMs, CUDA graphs, or Polar Express are used in this comparison.

The pinned, unmodified sources and hashes are recorded in
`muon_candidates.py`. They are fetched into ignored `runs/third_party/`;
no Python package is installed or upgraded. Research tests skip on a clean
clone until these optional sources are fetched.

Source revisions: [Microsoft Dion](https://github.com/microsoft/dion/tree/7692479288f928a3d61d1cc58137b4dbe4f71000),
[MuonPlus](https://github.com/K1seki221/MuonPlus/tree/8a9ace123afedaab8ba75ea0b19315594ae1da7c),
and [NorMuon](https://github.com/zichongli5/NorMuon/tree/c6989a8354730695d9f5a9faa6c55eeb24865209).
NorMuon's original source was inspected; this screen executes Microsoft's
batched implementation of the algorithm.

## Ownership and scope

All candidates change independent expert W1/W3/W2 matrices only. The expert
axis is never flattened into a matrix. Every other parameter, including
routers, gates, embeddings, attention/trunk matrices and norms, stays on
AdamW. This isolates expert optimization and matches the earlier Gefen
screen; it does not assess Muon on the whole trunk.

The research adapter requires a read-only expert store. Weights use the
existing pager, while optimizer history follows permanent expert UIDs in
host RAM. Only resident histories remain on the GPU. Missing projection
gradients and empty slots are skipped; resident zero gradients still count
as updates. Topology changes and writable checkpoints are rejected.

The AdamW control retains the trainer's existing slab-global bias-correction
clock; Gefen uses its qualified per-expert counters. The Muon-family matrix
rules have no Adam-style bias correction, and their momentum/row statistics
pause while evicted. This comparison includes those integration semantics,
rather than claiming to isolate only the published optimizer equations.

Its lossless `state_dict` is an experimental optimizer snapshot, not a
production directory checkpoint: model weights and pager contents must be
copied separately. There is deliberately no new trainer or expert-file
format. A future integration requires its own persistence qualification.

## Learning protocol

The downloaded undertrained checkpoint is opened read-only. All candidates
start with fresh history, including subsequently admitted experts. The
architecture is hidden 512, expert width 2048, 32 resident experts, fixed
depth 6, context 512, and gradient clipping at norm 1. All touched experts
remain cached in host RAM. No learned checkpoint is written.

The locally available TinyStories validation text provides disjoint byte/character-token
ranges: adaptation `[0,262144)`, tuning from 262144, and test from 524288.
Evaluation uses sixteen 512-token windows. Training windows are sampled
without replacement in a seeded permutation. These seeds change window order.
The paged pool keeps its default selection temperature of zero, so routing
uses deterministic top-k; the recorded `routing_seed` fields are all null.
The optimizer RNG-preservation test also guards future stochastic use, but
this experiment does not test stochastic routing. Original checkpoint/data
overlap is unknown; these are not canonical document-deduplicated splits.

Final learning-rate selection uses 256 updates on seed 42. Every candidate
gets `3e-6, 3e-5, 1e-4`; a winning edge is extended by approximately a factor
of three until it is bracketed by worse tested rates, subject to a bounded
search. AdamW also gets a zero-expert-rate control, measuring auxiliary-only
adaptation. Zero is an explicit boundary, not a new optimizer. Selection uses
only tuning cross-entropy. Confirmation then runs each selected rate for 256
updates on seeds 43/44. Two data-order seeds cannot establish a general quality
ranking or statistical equivalence.
The auxiliary-only zero-expert-rate control also runs on both confirmation
seeds, without participating in another rate-selection round.

### Reproducibility qualification

The initial exploratory sweep used 64-update tuning and a 256-update pilot.
Matching seeds and inputs did not make its training reproducible: backward
passes on the same fixed graph changed gradients under default GPU execution.
The exploratory learning rankings are **not used to choose an optimizer**.
Their raw outcomes are retained separately.

`compare_muon_deterministic.py` controls execution in each subprocess:

- `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; TF32 is disabled.
- The existing Sonic token-gather reduction uses its measured, valid Triton
  configuration: `BLOCK_H=256`, `BLOCK_K=32`, four warps, four stages. This
  fixes process-local dispatch and never edits the installed package. This
  screen restricts it to hidden 512 and 32 resident experts.
- Two independent eight-update processes per candidate must have identical
  gradients, training traces, weights, optimizer history,
  host expert contents, slots, and initial/final evaluation results. All
  candidates must receive bit-identical initial gradients.
- Every seed-42 learning run checks its first gradient norm/objective against
  that oracle; LR `3e-5` runs check the full eight-update prefix.
  Cached results must match the measured source hashes and execution controls.

Deterministic mode removed the observed repeated-backward variation; fixing
the third-party reduction selection also makes evaluation use the same math.
The pinned reduction is separately checked against the PyTorch oracle for
outputs and every derivative, including zero scores, missing tokens and
unused experts; checkpoint replay must be bit-identical.

Auxiliary pool LR is fixed at `1e-4`, trunk LR at `2e-5`, auxiliary Adam
betas `(0.9,0.95)`, and weight decay 0.1. These rates do not change when
the expert LR is swept. This differs from the earlier Gefen screen, which
scaled auxiliary LR along with expert LR, so results should not be mixed.

Muon-family optimizers are warmed up on synthetic weights at the actual
resident geometry before training. This compiles kernels without changing
checkpoint weights or the model/data RNG. Warmup time is separately recorded;
external telemetry includes warmup. Training timing excludes this warmup and
held-out evaluation, but includes paging, state transfers, and actual
optimizer calibration. Per-step traces also record synchronized optimizer
wall time. This is an adaptation screen, not a pretraining convergence proof.

## Reproduce

Run from the repository root with its existing `.venv` and qualified
Torch/Sonic/Gefen stack. The source setup only clones pinned Git repositories.

```sh
.venv/bin/python -m replication.muon_candidates
TORCHDYNAMO_DISABLE=1 .venv/bin/python -m unittest tests.test_muon_candidates -v
```

CUDA qualification, compilation, and warmup are externally monitored:

```sh
PATH="$PWD/.venv/bin:$PATH" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TORCHINDUCTOR_COMPILE_THREADS=2 \
.venv/bin/python -m replication.gpu_monitor \
  --out runs/muon-cuda-tests.power.json -- \
  .venv/bin/python -c '
import torch, unittest
import tests.test_muon_candidates as tests
torch.use_deterministic_algorithms(True)
torch._dynamo.config.recompile_limit = 64
tests.MuonCandidateTests.device = "cuda"
unittest.main(module=tests, argv=["controlled-cuda"], verbosity=2)
'
```

The qualification process raises its own Dynamo specialization limit to 64
because it intentionally covers many matrix shapes, missing-gradient cases,
and candidate algorithms in one process. No system configuration changes.

```sh
.venv/bin/python -m replication.run_muon_screen --phase repro \
  --out runs/muon-deterministic-repro-fixed
.venv/bin/python -m replication.run_muon_screen --phase long-grid
.venv/bin/python -m replication.run_muon_screen --phase long-confirm
.venv/bin/python -m replication.run_muon_screen --phase latency
.venv/bin/python -m replication.run_muon_screen --phase latency-large
.venv/bin/python -m replication.summarize_muon_screen
```

`long-confirm` includes both matched auxiliary-only controls. Repeating the
learning phases reuses only results that still pass the source/protocol
checks; timing phases require a fresh output location.

The runner monitors every GPU subprocess externally. To repeat the standalone
fixed-reduction qualification:

```sh
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m replication.gpu_monitor \
  --out runs/muon-fixed-reduction.power.json -- \
  .venv/bin/python -m replication.probe_fixed_sonic_reduction
```

The older short-screen modes remain available for experiments, but are not
the final learning-selection protocol. `compare_muon_training.py` preserves
the original uncontrolled experiment; the runner uses the controlled module.

The runner serializes GPU workloads and starts an external NVML monitor for
each subprocess. Raw timestamped telemetry includes board power, utilization,
clocks, temperature and clock-limiting masks. Training summaries integrate
samples over the explicit training interval. Energy is board energy including
desktop activity, not whole-machine energy. Sampled peaks and short intervals
have sensor-resolution limitations. The external monitor avoids GIL sampling
gaps during Gefen's Python calibration.

`latency` measures eight independent 512/2048 experts (24 matrices,
25,165,824 parameters) with identical fixed correlated gradients for five
seconds after warmup, with deterministic execution controls. This excludes model computation and paging; it isolates
available optimizer implementations. Constant-gradient timing does not
establish learning quality.

`latency-large` repeats the optimizer-only probe at hidden 4096, expert width
16384, four experts (805,306,368 parameters). The allocator is capped at 75%
of GPU memory in both probes. An allocator OOM is recorded as a feasibility
result; other failures stop the comparison. This checks optimizer overhead
at a prospective native-FP4 geometry without claiming that a complete
recurrent training step fits with the same headroom.
Reported optimizer-only peak allocated memory covers the timed period after
warmup. The allocator cap applies during warmup too; external NVML records
the complete process and includes allocations outside PyTorch's allocator.

## Results

Measured on this RTX 5090 with the existing repository environment. The final
controlled study contains **43 complete 256-update learning trials**
(27 tuning/control trials plus 14 selected-rate confirmations and two matched
auxiliary-only confirmations), 14 independent eight-update replay runs, and
14 optimizer-only timing probes. Uncontrolled exploratory learning outcomes
are excluded from these tables.

### Held-out learning

Lower predictive cross-entropy is better; units are nats per byte/marker token.
Initial test CE is **1.45501** for every trial.
Rates were selected on seed-42 tuning loss only. The following are the two
additional data-order seeds, using those fixed rates.

| Expert optimizer | Selected LR | Test CE, seed 43 | Test CE, seed 44 | Mean test CE |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 3e-06 | 0.65426 | 0.58851 | 0.62138 |
| Gefen | 3e-05 | 0.64871 | 0.58997 | 0.61934 |
| Muon | 3e-06 | 0.70733 | 0.58614 | 0.64673 |
| Muon+ | 0.0001 | 0.64151 | 0.59270 | 0.61711 |
| NorMuon | 3e-06 | 0.68371 | 0.60191 | 0.64281 |
| Dion3 0.25 | 3e-05 | 0.62699 | 0.60801 | 0.61750 |
| Dion3 0.50 | 1e-06 | 0.66970 | 0.60675 | 0.63822 |
| Experts frozen; auxiliary AdamW | 0 | 0.71075 | 0.61065 | 0.66070 |

Muon+ and Dion3-quarter have nearly identical means; neither beats AdamW
on both confirmation seeds. Gefen also changes order relative to AdamW.
NorMuon and Dion3-half show no quality advantage here. These results do
not establish a general optimizer ranking, statistical equivalence, or
pretraining convergence. The frozen-expert control shows substantial
adaptation from the common auxiliary optimizer alone.

### Monitored adaptation cost

Means across seeds 43/44, 131,072 new model tokens per run. Timing includes
paging, host history transfers, and first-use calibration; it excludes
synthetic optimizer warmup and held-out evaluation. Peak watts and VRAM
are the larger recorded peak across those two runs.

| Expert optimizer | Train s | New tokens/s | Avg W | Peak W | J/new token | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 41.04 | 3195 | 155.1 | 169.0 | 0.04855 | 2.293 |
| Gefen | 46.95 | 2792 | 172.8 | 190.7 | 0.06191 | 1.638 |
| Muon | 34.56 | 3794 | 215.7 | 230.3 | 0.05687 | 2.407 |
| Muon+ | 37.44 | 3501 | 198.6 | 212.2 | 0.05673 | 1.918 |
| NorMuon | 35.11 | 3733 | 219.1 | 234.9 | 0.05869 | 2.407 |
| Dion3 0.25 | 34.04 | 3850 | 185.0 | 198.3 | 0.04804 | 2.033 |
| Dion3 0.50 | 35.05 | 3741 | 195.9 | 210.8 | 0.05237 | 2.064 |

Gefen's final-16-step mean is 17.5 ms for the optimizer
and 104.1 ms for the full step; Dion3-quarter's is
54.6 / 128.8 ms. Gefen's complete short run
is slower despite its faster tail. Its first optimizer step alone takes
3.85 seconds in each run and includes calibration. These tail
windows are descriptive, not a separate matched steady-state throughput
benchmark: learning-dependent routing changes the work. Do not extrapolate
the complete 256-update average to a long training run.

### Optimizer-only scaling

These probes use the deterministic execution controls, FP32 parameters and
gradients, and identical fixed correlated gradients. Five seconds are timed
after warmup/settling. Energy per step uses the measured elapsed interval,
not median latency multiplied by watts.

**Eight 512/2048 experts: 25.17 million parameters.**

| Optimizer | Median ms/step | Avg W | Peak W | J/step | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 0.506 | 367.9 | 376.6 | 0.187 | 0.375 |
| Gefen | 0.480 | 475.2 | 484.7 | 0.229 | 0.211 |
| Muon | 4.414 | 595.5 | 600.5 | 2.632 | 0.539 |
| Muon+ | 6.168 | 346.3 | 349.5 | 2.138 | 0.328 |
| NorMuon | 4.883 | 593.0 | 600.3 | 2.899 | 0.539 |
| Dion3 0.25 | 3.177 | 336.4 | 340.5 | 1.070 | 0.446 |
| Dion3 0.50 | 4.153 | 452.0 | 457.3 | 1.879 | 0.453 |

**Four 4096/16384 experts: 805.31 million parameters.**

| Optimizer | Median ms/step | Avg W | Peak W | J/step | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 14.911 | 382.9 | 388.4 | 5.715 | 12.000 |
| Gefen | 13.487 | 521.8 | 532.5 | 7.049 | 6.753 |
| Muon | 406.957 | 600.1 | 602.3 | 244.271 | 16.281 |
| Muon+ | 433.340 | 599.9 | 600.5 | 259.906 | 9.812 |
| NorMuon | 425.941 | 600.3 | 605.3 | 255.687 | 16.282 |
| Dion3 0.25 | 126.444 | 589.9 | 590.9 | 74.603 | 13.282 |
| Dion3 0.50 | 218.015 | 598.2 | 603.6 | 130.437 | 13.532 |

All seven optimizer probes fit the 75% allocator cap at both geometries.
This does not prove full-model training fits: model computation, activation
storage, paging, and more resident experts are absent from these probes.
The large Muon-family probes reach roughly the board power limit while
taking much longer than AdamW/Gefen. Higher power is not a throughput gain.
Gefen uses less memory than AdamW here but more board energy per isolated
step; its memory, speed and energy tradeoffs are separate measurements.

### Decision and limits

**Carry Gefen and Dion3-quarter into the next longer learning study, with
AdamW retained as the reference.** Gefen has the strongest measured memory
and wide-matrix latency case. Dion3-quarter is the most practical tested
Muon-family candidate: its held-out mean is close to Muon+ and its measured
cost is substantially lower. This is a shortlist, not a production optimizer
selection. The trainer still exposes only its existing AdamW/Gefen choices.

The large-geometry probe also makes orthogonalization an explicit cost to
resolve before selecting a Muon-family optimizer for fast NVFP4 experts.
These available implementations and classic/Keller NS5 settings do not
represent every possible optimized Muon implementation. Gram NS, Polar
Express, other momentum settings, trunk-wide Muon, and native NVFP4
forward/backward quality remain untested. No result here supports claiming
that low-precision learning quality is already qualified.

A next quality experiment needs longer runs, more seeds and a defensible
document-level data split, while retaining the higher-precision reference.
The measured checkpoint, depth-6 adaptation and tiny held-out ranges are
the limits of the current quality evidence.

### Verification and retained evidence

- **109 tests passed** in the complete regression suite; all six candidate
  qualification tests also passed separately on CUDA with deterministic
  controls. Those tests cover all five matrix configurations, bit-identical
  paged/unpaged updates and state, eviction/swaps, missing/zero gradients,
  empty slots, closure paging, lossless NPZ snapshot/resume and RNG preservation.
- Independent-process replay passed for all seven configurations; all seven
  shared-reference-rate 256-update runs exactly matched the eight-update
  oracle prefix. This proves the stated replay checks, not every future
  GPU shape or an independently repeated full 256-update trajectory.
- The fixed Sonic reduction passed output and all-gradient comparisons
  against the PyTorch reference, plus bit-identical checkpoint replay,
  zero-score derivatives, missing tokens and unused experts.
- All **401 original checkpoint files** retained identical sizes and SHA-256
  hashes. Learned weights remained in RAM. No package, driver, service,
  clock or power-limit configuration changed.

Timestamped external NVML samples include compilation and warmup. The
largest confirmation-training sample gap is 0.102 seconds.
Sensor averaging and desktop activity limit power/energy precision; sampled
peaks are not guaranteed instantaneous peaks. These timings describe the
controlled research harness, not a promised production-trainer speedup.

- [Controlled results and complete rate grids](results/muon_variants.json)
- [Verification, environment, archive checksum](results/muon_variants_verification.json)
- [Raw telemetry, traces, replay gates, source snapshots and historical diagnostics](results/muon_variants_evidence.tar.gz)
- [Excluded exploratory aggregate, explicitly unqualified](results/muon_variants_exploratory.json)

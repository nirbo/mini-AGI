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

The locally available TinyStories validation text provides disjoint token
ranges: adaptation `[0,262144)`, tuning from 262144, and test from 524288.
Evaluation uses sixteen 512-token windows. Training windows are sampled
without replacement in a seeded permutation. Original checkpoint/data
overlap is unknown; these are not canonical document-deduplicated splits.

The expert learning-rate grid is `3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2`,
64 updates each, data-order seed 42. Select using tuning cross-entropy
only. Then run every candidate for 256 updates on data-order seeds 43/44.
Check whether the grid winner lies at an edge before accepting selection.
The two lowest positive rates and an AdamW expert-LR-zero control were added
after the first two rates showed that smaller expert steps could help.
The zero-rate control measures auxiliary-only adaptation; it is not a new
optimizer. Rate selection refuses an unbracketed winner.

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
PATH="$PWD/.venv/bin:$PATH" MINAGI_MUON_TEST_CUDA=1 \
TORCHINDUCTOR_COMPILE_THREADS=2 \
.venv/bin/python -m replication.gpu_monitor \
  --out runs/muon-cuda-tests.power.json -- \
  .venv/bin/python -m tests.test_muon_candidates -v
```

The qualification process raises its own Dynamo specialization limit to 64
because it intentionally covers many matrix shapes, missing-gradient cases,
and candidate algorithms in one process. No system configuration changes.

```sh
.venv/bin/python -m replication.run_muon_screen --phase smoke
.venv/bin/python -m replication.run_muon_screen --phase grid
.venv/bin/python -m replication.run_muon_screen --phase low-grid
.venv/bin/python -m replication.run_muon_screen --phase confirm
.venv/bin/python -m replication.run_muon_screen --phase latency
.venv/bin/python -m replication.run_muon_screen --phase latency-large
```

The runner serializes GPU workloads and starts an external NVML monitor for
each subprocess. Raw timestamped telemetry includes board power, utilization,
clocks, temperature and clock-limiting masks. Training summaries integrate
samples over the explicit training interval. Energy is board energy including
desktop activity, not whole-machine energy. Sampled peaks and short intervals
have sensor-resolution limitations. The external monitor avoids GIL sampling
gaps during Gefen's Python calibration.

`latency` measures eight independent 512/2048 experts (24 matrices,
25,165,824 parameters) with identical fixed correlated gradients for five
seconds after warmup. This excludes model computation and paging; it isolates
available optimizer implementations. Constant-gradient timing does not
establish learning quality.

`latency-large` repeats the optimizer-only probe at hidden 4096, expert width
16384, four experts (805,306,368 parameters). The allocator is capped at 75%
of GPU memory in both probes. An allocator OOM is recorded as a feasibility
result; other failures stop the comparison. This checks optimizer overhead
at a prospective native-FP4 geometry without claiming that a complete
recurrent training step fits with the same headroom.

## Results

Results and verification evidence are populated after the complete screen.

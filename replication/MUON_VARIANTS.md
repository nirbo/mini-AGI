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

Results and verification evidence are populated after the complete screen.

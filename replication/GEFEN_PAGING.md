# Optional Gefen expert paging

The adapter uses Gefen 1.2.3 for each independent expert W1/W3/W2 matrix and
AdamW for all other parameters, retaining FP32 masters and gradients. AdamW
remains the default. The implementation is verified against stock Gefen;
learning/performance evidence below is a limited BF16 adaptation experiment,
not a pretraining or native-FP4 quality qualification.

## Ownership, clocks and storage

`minagi/optimizers.py` keeps the model's three fixed slabs as autograd
parameters. Independent 2D parameter views share their storage for Gefen
updates. The expert axis is never flattened into one matrix. The auxiliary
AdamW step excludes those slabs, so an expert is updated exactly once.
Cache generations are invalidated through the existing pool post-step hook.

Each expert/projection counter advances when its resident slab has a gradient,
including a zero gradient, and pauses while the expert is evicted. A missing
slab gradient and an empty slot are skipped. This is deliberately different
from the existing AdamW slab-global clock; the learning comparison evaluates
the two integrations, not only the optimizer formulas. Closures may change
admission; state is restored after the closure, before the update.

Expert state is serialized by permanent UID alongside its weights in
`experts/eNNNNN.npz`. Reserved `__gefen_` arrays contain a versioned JSON tree
and typed tensor storage: uint8 momentum indices, FP32 magnitudes/second
moments, partition and shape metadata, counters and codebook fingerprint.
No pickle is used. Encoded tensors are never narrowed to BF16 or converted
through FP32 when paging. Incoming identity, geometry, partition, tensor
types/shapes and codebook identity are checked before update kernels launch.

The shared learned codebook is immutable under this Gefen implementation.
It is atomically written and fsynced to `experts.gefen-codebook.npz` before
any trained expert can be written out on RAM eviction. The companion is
loaded when the optimizer is constructed. This prevents orphaned compressed
history before the first regular checkpoint. It does not turn the existing
live weights directory into a transactional snapshot after arbitrary crashes.

`optim.npz` stores versioned, lossless auxiliary AdamW state, named parameter
groups and the exact Gefen global counter/codebook. Full checkpoint resume
restores this state and gets each expert's own state from its file. Copy the
entire weights directory, including the companion file, to snapshot it.
`optimizer.state_dict()` carries resident UID-owned states; nonresident state
continues to belong to the expert store, as with the existing pager.

Growth/pruning keep auxiliary parameters on AdamW, so existing row-wise state
carry works for gates and routers. Newly born expert matrices start with fresh
Gefen history. UID remapping during pruning does not rename expert history.
The model must already be on its target device before optimizer construction;
moving/replacing the expert slab storage afterwards is rejected. AMSGrad,
maximize, capturable and differentiable expert groups are unsupported and
rejected. Masters must be FP32.

First adoption of Gefen from an Adam checkpoint restores legacy auxiliary
moments but starts fresh expert history; it is not an Adam-to-Gefen momentum
conversion. Switching back to AdamW without explicitly resetting history is
rejected. Incompatible/corrupt Gefen history propagates an error rather than
silently starting over. Raw checkpoint APIs restore group options exactly;
reader/live launch APIs then apply the caller's LR and weight decay, matching
their established behavior. A resized or different parameter set requires an
explicit migration, not a silent partial restore.

## Use and environment

`train.py read ... --optimizer gefen` enables the adapter. Configuration also
accepts `training.optimizer: gefen`. `serve.py --optimizer gefen` and
`LiveLearner(..., optimizer="gefen")` support continuing learning from these
checkpoints. Read-only inference needs no Gefen optimizer instance.

Only the repository `.venv` was used. No dependency was installed or upgraded
during this integration task, and no system, driver, clock or power-limit
setting changed. The already qualified stack is Torch 2.14.1+cu130,
Triton 3.8, QuACK 0.6.5, source-patched SonicMoE 0.1.2.post1 and Gefen 1.2.3
at `e735f851a7ebdf0106da87126458c86e3e583d99` (Numba 0.68, llvmlite 0.50).

Gefen's source preflight rejects toolkit 13.1 versus Torch CUDA 13.0. The
retained `patches/gefen-cuda-minor-opt-in.patch` permits this same-major mismatch
only with `MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH=1`; it changes no optimizer
arithmetic. The existing CUDA 13.1.115/GCC 15.2 setup successfully builds it.
No toolkit installation or system change is required on this machine.
Set `PATH` to include the main `.venv/bin` so its Ninja executable is found.

Example qualification from this branch/worktree:

```sh
PATH="/home/nir/ml-tools/mini-AGI/.venv/bin:$PATH" \
MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 \
QUACK_CACHE_DIR=/home/nir/ml-tools/mini-AGI/runs/dependencies/quack-expert-backends-test \
/home/nir/ml-tools/mini-AGI/.venv/bin/python -m replication.gpu_monitor \
  --out /tmp/gefen-suite.power.json -- \
  /home/nir/ml-tools/mini-AGI/.venv/bin/python -m unittest discover -s tests -v
```

## Verification

**96 tests passed**, including the existing reference/Sonic causality and
cache regressions, plus CPU and fused CUDA versions of the Gefen tests:

- Paged eviction/disk reload versus uninterrupted stock Gefen, with
  bit-identical weights, compressed state and codebook; closures that page;
  zero gradients, missing projection gradients and empty slots.
- Growth, pruning and full directory resume into different physical slot
  order; UID-owned state, auxiliary moments and counters remain exact.
- Lossless NPZ/state-dict round trips, including all 256 uint8 values and
  BF16 metadata tensor support in the codec.
- Mismatched UID, geometry, partition, index tensor and codebook rejection
  before weight updates; recovery after valid state is restored.
- Recoverable evicted history before any `optim.npz` checkpoint exists.
- Legacy Adam adoption, recurrent forward/backward and compute-cache refresh.
- Live-learning resume, requested LR/decay overrides, one live update and save.

All GPU tests, compilation and warmup were power monitored. The final suite
took 51.6 seconds; average sampled board power was 277 W and peak 324 W.
These whole-process numbers are verification telemetry, not kernel benchmarks.
An initial failed run lacked `.venv/bin` on `PATH` for Ninja and had a CPU/GPU
device mistake in a test helper; both are corrected and failure logs retained.

## Learning protocol and limits

`compare_gefen_training.py` loads the downloaded checkpoint read-only. Both
candidates start with fresh optimizer history, including experts paged in
later. All touched experts remain cached in host RAM; no learned checkpoint
is written. Tests use temporary synthetic writable checkpoints, then clean up.

Compute is Sonic BF16 at hidden 512/expert width 2048, 32 resident experts,
context 512, fixed depth 6, gradient norm clipping 1, betas (0.9, 0.95), decay
0.1, trunk LR 0.2 times pool LR. The original checkpoint depth is 24; this
screen does not qualify that original workload or native FP4 learning.

Adaptation samples nonoverlapping windows from the first 262,144 tokens of
the locally available TinyStories validation text, in a seeded permutation.
Tuning and test use sixteen 512-token windows beginning at offsets 262,144
and 524,288. These are character/byte-token partitions, not a canonical
training split or a document-deduplicated benchmark; original checkpoint
training overlap is unknown. Comparisons use the same data order and exactly
equal initial held-out results for each pair. Test loss is reported per run;
LR selection uses only tuning loss on seed 42.

The 64-update learning-rate grid is 3e-6, 1e-5, 3e-5, 1e-4, 3e-4 and 1e-3.
Both candidates select **1e-4**, with worse tuning loss at higher grid values.
Two additional data-order seeds, 43/44, confirm the selected rate and are
extended to 256 updates. The longer horizon reuses that rate rather than
retuning it. Two confirmation seeds and sixteen test windows are insufficient
to establish statistical equivalence or a reliable convergence advantage.

Training time includes first-step calibration, new-expert partition inference,
paging and Python/launch overhead; initial/final evaluation is excluded.
Selected-rate and longer runs also have an external power-monitor process.
This avoids the internal monitor's roughly two-second GIL sampling gaps during
initial Gefen calibration. Both raw records are retained; comparisons use the
external records. Reported energy is GPU board energy, not whole-system energy.
Short sensor windows and desktop activity still limit precision.

At 64 updates, mean test CE across seeds 42/43/44 is 0.7475 for AdamW and
0.7554 for Gefen; results vary in direction by data order. Peak allocated VRAM
is **2.293 GiB versus 1.638 GiB**, a 0.655 GiB saving. This supports memory
efficiency without establishing a time-to-quality win.

At 256 updates (131,072 training tokens), the two independent confirmation
seeds give the following results at the selected rate. Time and energy include
calibration and all updates; each row aggregates both runs.

| Optimizer | Mean test CE | Mean training s | Tokens/s | Average / peak sampled W | Board J/token | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 0.6605 | 40.51 | 3236 | 161 / 322 | 0.04965 | 2.293 |
| Gefen | **0.6191** | 46.74 | 2804 | 178 / 322 | 0.06337 | **1.638** |

Gefen has lower final test CE in both confirmation runs (differences -0.0277
and -0.0552 nats), about 29% less peak allocated memory, 15% more training time
and 28% more board energy per token. The first-step/new-expert calibration
cost amortizes: its overall time penalty falls from roughly 115% at 64 updates
to 15% at 256. Mean last-sixteen full-step times are 150 ms for AdamW and
98 ms for Gefen, but learned routing/work changes with the optimizer, so this
is not an isolated kernel speed comparison. The experiments measure fixed
token budgets and endpoints, not a dense time-to-target-loss curve.

This is enough to qualify an **optional** integration with demonstrated memory
savings and promising short-run learning. It does not justify replacing the
default, asserting a general quality advantage, or assuming the same outcome
with native FP4. Next qualifications are realistic bounded-host-cache I/O,
longer/more diverse learning runs and native-FP4 gradient stability.

Full learning and power measurements are in
[`results/gefen_paging.json`](results/gefen_paging.json), with raw results,
timestamped telemetry, logs and verified SHA-256 manifest in
[`results/gefen_paging_evidence.tar.gz`](results/gefen_paging_evidence.tar.gz).
`summarize_gefen_paging.py` assembles this evidence without GPU execution.

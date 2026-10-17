# Trainer guide

This guide describes the trainer in the fork's `dev` branch. Start with
`train.py read` for ordinary text and paged experts. Settings live in
[config.yaml](config.yaml); the command implementations are in
[train.py](train.py). Some CLI help and older README examples describe an
earlier reader. The behavior and qualifications below follow the current code.

- [Environment and commands](#environment-and-command-layout)
- [Reader behavior and persistence](#what-read-does)
- [Reader examples](#practical-read-examples)
- [Reader option reference](#read-option-reference)
- [Packed streaming](#packed-corpus-stream)
- [Depth probe](#depth-diagnostic-ponder-probe)
- [Monitoring and troubleshooting](#monitoring-comparisons-and-troubleshooting)

## Environment and command layout

Run commands from the repository root using its `.venv`. On this desktop that
is `/home/nir/ml-tools/mini-AGI`. Create the environment only if missing:

```bash
if [ ! -d .venv ]; then
    uv venv .venv
fi
export PATH="$PWD/.venv/bin:$PATH"
.venv/bin/python --version
```

The existing desktop environment already has the qualified dependencies.
Do not reinstall or upgrade them just to run these examples. In a new
environment, install Python dependencies with
`uv pip install --python .venv/bin/python ...`, never into the system Python.
The basic trainer needs PyTorch, NumPy and PyYAML; plotting adds Matplotlib.
Corpus builders and optional kernels have additional dependencies. The tested
5090 stack and source patches are recorded in
[EXPERT_BACKENDS.md](replication/EXPERT_BACKENDS.md) and
[GEFEN_PAGING.md](replication/GEFEN_PAGING.md).

```bash
.venv/bin/python train.py --device cpu --help
.venv/bin/python train.py --device cpu read --help
.venv/bin/python train.py --device cpu stream --help
.venv/bin/python train.py --device cpu ponder-probe --help
```

`--device` is a **global option: put it before the subcommand**. Use `cuda`,
`cuda:0`, or `cpu`. If omitted, the trainer selects CUDA when available,
otherwise CPU. It does not automatically retry training on CPU after a CUDA
out-of-memory error.

| Command | Input | Training behavior | Expert storage / optimizer |
| --- | --- | --- | --- |
| `read` | UTF-8 text files or directories | Batch 1; re-forwards and learns the full sliding window every update | Disk/RAM/VRAM paging; AdamW or optional Gefen experts + AdamW auxiliaries |
| `stream` | Packed `train.bin` / `val.bin` source directories | Batch 1; cached consecutive chunks; gradient stops at chunk boundaries; optional accumulation/replay | Fully resident pool; AdamW |
| `ponder-probe` | A checkpoint and arithmetic prompts | No optimizer steps; reports halting depth versus problem size | Loads the checkpoint's architecture |

## What `read` does

- Recursively collects text, skipping known binary extensions, hidden/cache
  directories and samples that fail UTF-8/printability checks. No trained
  tokenizer or packed corpus is required. File verdicts are cached in
  `runs/corpus_index.json`.
- Groups files into subject lanes, primarily by top-level directories under
  each supplied root. Subjects rotate; each visit samples a file and offset,
  then advances contiguously through a passage. Large subjects do not
  automatically receive proportionally more visits.
- Advances the cursor by up to `--chunk` tokens, then forwards, scores and
  backpropagates through the last active context window. Overlapping text is
  learned again. **There is no detached KV cache between learning steps on
  this path.** Longer context increases training work and activation memory.
- Admits experts causally from first-pass prefix requests, keeps them stable
  through backward/update, and pages weights and expert-owned history through
  the RAM cache. Optional capacity limits drop assignments causally.
- Supports adaptive halting, sampled training depth, recurrent BPTT limits,
  expert activation recomputation, context growth and expert growth/pruning.
- Evaluates predictive held-out loss, writes samples/telemetry, and can adjust
  learning rate from held-out evidence through the plasticity controller.

“Characters” in flags/logs usually means byte-token positions, not Unicode
code points. The vocabulary is 256 bytes plus nine structural markers;
`<user>`, `<think>` and `<|endoftext|>`, for example, each occupy one token.
Corpus-size budgets start from file bytes, while the reader's progress counter
adds configured chunks, including short tails. Treat progress/pass counts as
approximate reading budgets, not exact counts of unique training tokens.

### Configuration and resume

`config.yaml` is resolved relative to the repository, not selected through a
`--config` flag. Explicit CLI options override the defaults they expose.

On first creation, model dimensions and initial pool shape come from the
configuration. On resume, learned tensor shapes and the depth ceiling come
from the checkpoint. Editing `model.d_model`, head count, expert width/depth,
or block counts does not resize an existing model. There is no `read` CLI flag
for changing its `max_steps` ceiling. Runtime overrides cover context, resident
slots, RAM capacity, compute precision, optimizer, training-depth/halting
policy and the exposed growth controls. Pool balance, selection temperature
and capacity factor are read from YAML at load.

The checked-in YAML currently sets hidden width 512, expert width 2048,
64 initial experts/top-8, 32 resident slots, RAM cache 96, context 2048→4096,
chunk 2048, up to 24 recurrent rows, BF16 compute, LR `3e-4`, and trunk LR
multiplier `0.1`. A loaded checkpoint can have more experts and a different
shape. These values describe configuration, not a guaranteed fit/performance
preset for either a 16 GB or 32 GB GPU.

Resume restores weights, optimizer history, cumulative counters, context and
plasticity state where supported. File sampling incorporates the checkpoint's
read count rather than restarting the same random sequence. It does not
restore an exact file cursor or guarantee bit-identical continuation of the
whole stochastic training/data process. `--passes N` budgets approximately
`N × corpus bytes` of reading **in this invocation**, sampling with replacement;
it does not guarantee every file or offset is visited exactly once.

### Persistence and stopping

`--save` enables explicit checkpoints and permits expert growth/pruning.
For an existing model, omitting it skips those explicit saves, **but is not a
read-only guarantee**: `read` constructs a writable pager, so trained experts
can be written back when the RAM cache evicts them. Gefen can also write its
durable codebook companion. If weights are absent, the reader creates a model
on disk even without `--save`. A manifest-only directory is replaced during
creation; a partial/nonempty foreign directory without `core.npz` is rejected.
Use a separate copy of a checkpoint for experiments you intend to discard.

With saving enabled, checkpoints are written on the timer, sample rounds,
topology changes, graceful stops and normal completion. `--save-every 0`
disables the timer only; it does not prevent the other saves. `read` saves the
current model, rather than maintaining a separate best-checkpoint snapshot.

The weights directory contains `manifest.json`, `core.npz`, `routers.npz`,
`optim.npz`, and `experts/eNNNNN.npz`. Gefen adds
`experts.gefen-codebook.npz`. Copy the **whole directory** to preserve history
and metadata. Files are replaced atomically individually; the directory is a
live store, not a transactionally frozen snapshot. Stop its writer before
copying it, and give experiments separate model and log paths.

During the training loop, the first Ctrl-C/SIGTERM requests a stop after the
current chunk, with a checkpoint if `--save` is set. A second Ctrl-C interrupts
immediately. `--minutes` is checked between chunks after startup; startup,
evaluation and final sampling can make wall time exceed that value.

## Practical `read` examples

Replace corpus paths with your own **separate training and validation data**.
Only point `--weights-dir` at a model you intend this run to own. Examples are
usage templates, not benchmarked GPU memory presets.

### Start a fresh model

```bash
.venv/bin/python train.py --device cuda read /path/to/corpus/train \
  --weights-dir runs/fresh-model \
  --held-out /path/to/corpus/val \
  --save --sample-every 10 \
  --sample-log runs/fresh-samples.txt \
  --history runs/fresh-experts.jsonl
```

Creates `runs/fresh-model` from YAML if no model exists there, learns from the
training tree, checkpoints, and samples/evaluates about every ten minutes of
reading. Running it again resumes the existing model. An existing trained
directory is never turned into a fresh model just by changing YAML shape.

To build the repository's text corpus instead of supplying your own:

```bash
.venv/bin/python -m corpora all --limit 5000
```

Downloads/generates a smaller corpus under `data/train` and `data/val`; it
can still require additional dependencies and storage. Existing lanes are
normally skipped. See `.venv/bin/python -m corpora all --help` before changing
build scope. The trainer itself does not download weights or datasets.

### Copy the downloaded checkpoint, then run a bounded experiment

Use a new destination each time; the following assumes
`runs/trainer-demo/weights` does not already exist and the source has no writer:

```bash
mkdir -p runs/trainer-demo
cp -a --reflink=auto /mnt/fast/models/mini-AGI/undertrained/weights \
  runs/trainer-demo/weights

.venv/bin/python -m replication.gpu_monitor \
  --out runs/trainer-demo/power.json -- \
  .venv/bin/python train.py --device cuda read /path/to/corpus/train \
  --weights-dir runs/trainer-demo/weights \
  --held-out /path/to/corpus/val \
  --minutes 5 --save --sample-every 1 \
  --chunk 512 --context-end 2048 --resident 16 \
  --grow-k 0 --no-plots \
  --sample-log runs/trainer-demo/samples.txt \
  --history runs/trainer-demo/experts.jsonl
```

Copies the checkpoint (reflink when supported, normal copy otherwise), then
trains only the copy. It caps context, lowers physical residency, disables
both expert growth and pruning, and records board telemetry. It does not
change the checkpoint's width or depth ceiling. Copying can require the full
checkpoint's disk space. The RAM cache and activation/depth costs still matter.

### Continue with Gefen and/or Sonic

```bash
mkdir -p runs/trainer-gefen
cp -a --reflink=auto /mnt/fast/models/mini-AGI/undertrained/weights \
  runs/trainer-gefen/weights

MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 \
MINAGI_EXPERT_BACKEND=sonic \
.venv/bin/python train.py --device cuda read /path/to/corpus/train \
  --weights-dir runs/trainer-gefen/weights \
  --held-out /path/to/corpus/val --save \
  --optimizer gefen --precision bf16 --lr 1e-4 \
  --trunk-lr-mult 0.2 --sample-every 5 \
  --sample-log runs/trainer-gefen/samples.txt \
  --history runs/trainer-gefen/experts.jsonl
```

Enables Gefen for independent expert matrices and AdamW for auxiliary
parameters, while Sonic supplies expert forward/backward arithmetic. These are
independent choices: set `MINAGI_EXPERT_BACKEND=reference` to use reference
compute, or use `--optimizer adamw` with Sonic. The LR is an example, not a
universal tuned recommendation. Apply the monitoring wrapper above to GPU runs.
This uses a separate new copy so the earlier AdamW examples remain resumable.

Gefen requires the qualified optional 1.2.3 package. The CUDA minor-version
opt-in shown is specific to this machine's recorded build setup; it changes
preflight checks, not arithmetic. First adoption from Adam restores auxiliary
history and starts fresh Gefen expert history. A Gefen checkpoint requires
Gefen to continue that history; switching optimizers is not an automatic state
conversion. See [the integration report](replication/GEFEN_PAGING.md).

Sonic requires its qualified source patch, CUDA BF16, depth-1 experts and
widths divisible by 16; the adapter accepts SM90/SM100/SM120 families, with the
RTX 5090 locally qualified. Unsupported configurations raise an error. See
[backend setup and limits](replication/EXPERT_BACKENDS.md). The `dev` trainer
does **not** expose FP8, NVFP4, NVFP4 QAT, Muon, BF16 master weights, multi-GPU/DDP
training or a `read --accum` option. BF16 compute is not BF16 master storage.

### Adjust depth/memory, or capture routes

```bash
.venv/bin/python train.py --device cuda read /path/to/corpus/train \
  --weights-dir runs/trainer-demo/weights \
  --held-out /path/to/corpus/val --save --minutes 5 \
  --context-end 1024 --chunk 256 --resident 16 \
  --bptt-window 6 --train-steps-mean 3 --min-steps 1 \
  --grow-k 0 --sample-every 1 --no-plots \
  --sample-log runs/trainer-demo/depth-samples.txt \
  --history runs/trainer-demo/depth-experts.jsonl \
  --trace-routes runs/trainer-demo/routes.npz --trace-chunks 8
```

Limits the retained recurrent graph to six rows and samples depth as
`Poisson(3) + 1`, clipped to the allowed range. This changes the training
policy and may change quality; it does not resize the recurrent architecture.
The first eight traced forwards are written to NPZ with routing metadata.
The `tools/plot_routing.py` viewer named by older help is absent in this
checkout. Add `--no-pool-checkpoint` only to test the compute/memory tradeoff:
it disables expert recomputation and normally increases activation memory.

## `read` option reference

Defaults for most options come from YAML. This table groups the complete CLI;
`--help` prints accepted spellings, but the qualifications here take precedence
over stale descriptions. `-h` / `--help` exits without starting training.

### Data, optimization and persistence

| Options | Meaning / units |
| --- | --- |
| `paths ...` | Files/directories; defaults to `data.train` when omitted. |
| `--weights-dir` | Model directory; creation/resume and write target. |
| `--passes` | Approximate per-session corpus-size reading budget; default 1. |
| `--minutes` | Training-loop time limit in minutes; default 0 means no time limit. |
| `--chunk`, `--passage` | New byte tokens per update; contiguous byte-token passage per lane visit. Passage is rounded to whole chunks and at least one context window. |
| `--seed`, `--shuffle-seed` | Torch startup seed; file/offset sampling seed, also mixed with resumed read count. Both default 0. Fresh creation currently reseeds Torch to 0 internally rather than forwarding `--seed`. |
| `--optimizer` | `adamw` (default) or optional `gefen` hybrid. YAML key `training.optimizer` is supported. |
| `--lr`, `--trunk-lr-mult` | Base expert/pool LR and shared trunk multiplier. Actual LR also includes plasticity scale/warmup. |
| `--wd`, `--clip` | Weight decay and maximum global gradient norm. |
| `--lr-reset` | Reset the restored plasticity scale to 1; normal controller behavior still applies. |
| `--save`, `--save-every` | Explicit persistence; timer interval in minutes. See writeback and non-timer-save caveats above. |
| `--verbose` | Per-visit file/loss output. |

The `read` controller uses held-out evidence, not the `stream` cosine schedule.
Periodic observations happen in **sample rounds**, enabled with
`--sample-every > 0` and a valid held-out directory. Timer-only saves do not
collect validation observations. Deprecated YAML keys `training.lr_horizon`,
`lr_warmup` and `lr_floor` are rejected at startup, including for other trainer
subcommands because they share parser setup.

### Precision, context and recurrent compute

| Options | Meaning / units |
| --- | --- |
| `--precision` | `bf16`, `fp16`, `fp32`; compute precision. Masters and accumulated gradients remain FP32. FP16 is accepted but has no trainer GradScaler and is not a qualified substitute for BF16 here. |
| `--resident`, `--ram-capacity` | Expert count held in VRAM; expert count in the host LRU cache. Neither is a memory-size-in-GB flag. |
| `--context-end`, `--context` | Aliases for the maximum active window, in byte tokens; binds on resume and context growth. |
| `--context-start` | Initial active window when no saved `context_now` exists. Effective active context has a minimum of 64. |
| `--context-step`, `--context-gain-min` | Byte tokens per context growth; evidence threshold for continued growth. |
| `--context-every`, `--context-grow-every` | **Character/token budgets**, converted to step intervals using `--chunk`; help incorrectly labels these as chunks. Evidence collection and growth cadence respectively. |
| `--bptt-window` | Number of recurrent rows retained for backward, clamped to the checkpoint ceiling if too large. Distinct from token context. |
| `--train-steps-mean` | Poisson parameter for sampled training depth, then +1 and clipping; 0 disables sampling. |
| `--min-steps` | Minimum recurrent rows, clamped to `[1, max_steps]`. |
| `--ponder-beta`, `--halt-prior`, `--halt-thresh` | Halting KL weight, geometric prior, inference cumulative threshold. Current overrides use truthiness checks: passing numeric zero does not override a saved value. |
| `--halt-freeze`, `--no-halt-freeze` | Enable/disable skipping characters after halting. |
| `--no-pool-checkpoint` | Disable expert activation recomputation during backward; this is unrelated to saving model checkpoints. |

Context can be raised by rebuilding larger RoPE tables, but the trained model
does not automatically become proficient at that reach. Reducing context,
resident count or BPTT retention can reduce memory, with different costs to
learning. Reducing `chunk` alone does not bound `read`'s graph to that chunk.
The compute backend is selected with `MINAGI_EXPERT_BACKEND`, not a trainer
`--backend` flag. `MINAGI_DISPATCH=exact|padded` controls reference dispatch.

### Expert growth and pruning

| Options | Meaning / units |
| --- | --- |
| `--grow-k`, `--grow-every` | Experts to add per accepted decision; character/token budget between decisions. `--grow-k 0` disables the shared growth/pruning path. |
| `--prune-survival` | Character/token budget for newborn protection and subsequent unadmitted survival; converted to steps. Resident experts are protected from pruning. |
| `--grow-dying-frac`, `--dying-at` | Maximum allowed dying fraction for growth; fraction of the survival window at which an unadmitted expert counts as dying. |
| `--max-in-flight` | Maximum experts on newborn trial; 0 disables this brake. |
| `--grow-keep-ratio`, `--recent-mult` | Earning threshold for recent additions; multiplier defining the recent age range. |
| `--max-gap` | Growth brake from held-out minus current training-window **predictive CE**; 0 disables. Updated during sample evaluations, not from the regularized training objective. |
| `--max-disk-gb`, `--grow-mem-frac` | Expert-file growth budget in decimal GB (0 unlimited); GPU allocated-memory growth brake as a fraction. Not hard allocator or board power limits. |
| `--birth-gate` | Initial gate scale of new experts. |

These controls require `--save` to actually change expert files. Growth
requires the configured usage/earning/space brakes to permit it. Pruning is
based on time without admission, not simply a small gate. Without periodic
held-out observations the gap brake has no fresh gap evidence. There is no
`read --pool-max` flag; `stream` has its own expert-count ceiling.

### Evaluation, samples and records

| Options | Meaning / units |
| --- | --- |
| `--held-out` | Validation directory, reported per immediate subdirectory/domain. Missing/non-directory means no held-out evaluation. |
| `--eval-chars`, `--sample-eval-chars` | Approximate evaluation token budgets **per domain** for initial/final checks and sample rounds, converted to at least one chunk. Defaults are 122880 and 30720. |
| `--eval-every` | Accepted (default 4000) but currently unused by `read`. Use `--sample-every` for in-run evaluations. |
| `--sample-every`, `--sample-log`, `--sample-chars` | Minutes of reading between sample rounds (default 0 disables), log path, generated tokens per prompt/variant (default 140). Final sampling also runs when enabled. |
| `--sample-raw-only` | Generate just the raw greedy variant rather than raw plus repetition-adapted output. |
| `--adapt-strength`, `--adapt-decay`, `--rep-penalty` | Sample-generation repetition controls; they do not change the training loss. |
| `--history` | Expert telemetry JSONL; empty string disables. On resume, existing rows beyond the checkpoint's read count are removed. Default `runs/expert_history.jsonl`. |
| `--plots`, `--no-plots` | Redraw available plot tools after sample entries; output paths remain `runs/training_progress.png` and `runs/dashboard.png`. |
| `--plot-last`, `--plot-since` | Aliases: trailing window in **millions of characters** for the dashboard. |
| `--trace-routes`, `--trace-chunks` | NPZ route capture path and number of initial training forwards to retain (default 8). |

Held-out scores are causal halted-next-token predictive CE in nats/token;
lower is better. Logs also report bits/token (`CE / ln(2)`) and perplexity
(`exp(CE)`). Domain scores are combined with equal domain weight. Training
objective includes expected-depth CE, halting and pool penalties, so subtracting
it from held-out CE is not a valid generalization gap. Historical upstream
losses used a different objective and should not be compared as the same metric.
Reported evaluation uncertainty does not establish variation across training
seeds or rule out data contamination.

The sample log can include training objective, predictive CE, gradient norm,
plasticity scale, context, reading/writing rates and repetition statistics.
`GradSNR` is observed for the trunk; it is telemetry, not an LR controller.
`read` does not automatically emit the `stream` history file or a board-power
trace. Index caches, sample logs, route captures, plots and history can be
written even when explicit model checkpointing is disabled.

## Packed-corpus `stream`

Each source in `--mix` must contain `train.bin` and `val.bin`: native uint16
token IDs compatible with the 265-token byte vocabulary. Packed corpus
directories are different from `data/train/...` text trees. For example:

```bash
.venv/bin/python -m corpora arithmetic --n 10000 --val 1000
.venv/bin/python train.py --device cuda stream \
  --mix data_math_char:1 --weights-dir runs/packed-demo-weights \
  --out runs/packed-demo-log --steps 1000 \
  --context 4096 --ctx-start 1024 --chunk 256 --accum 4 \
  --eval-every 100 --eval-chunks 32 --grow-k 0
```

Generates a small arithmetic corpus, then runs 1000 total optimizer steps on
a new, fully resident model (assuming the weights destination is new).
Four consecutive chunks contribute to each update; each backward frees its
chunk's graph. KV caches carry context but are detached between chunks. Sources
are sampled by normalized mix weights at window boundaries; replay revisits
remembered offsets in bursts. Context ramps toward the ceiling early in the run.

`stream` does not call the YAML-driven fresh-model creator: it starts from
`RecurConfig` defaults plus its CLI overrides when no manifest exists. It
otherwise attempts to restore compatible weights/config and Adam history.
Its fixed `pool-max` can conflict with checkpoint router shapes; **do not
assume a paged or Gefen checkpoint is interchangeable with this path**.
Use `read` for the downloaded paged checkpoint and Gefen continuation.

| Options | Meaning / defaults |
| --- | --- |
| `--mix` | Comma-separated `directory:weight`; default `data_char:0.25,data_math_char:0.15,data_chat_char:0.25,data_chess_char:0.35`. |
| `--weights-dir`, `--out` | Best-model directory (default `weights`); logs directory (default `runs/agi`). |
| `--context`, `--ctx-start`, `--chunk`, `--accum` | Max context 32768, initial context 4096, chunk 512, accumulated chunks/update 1. Keep chunks compatible with context; effective update size is approximately `accum × chunk`. |
| `--steps` | Absolute loop ceiling, default 20000; resume begins at saved step +1, not 20000 additional steps. |
| `--lr`, `--trunk-lr-mult`, `--warmup` | Base LR `1e-4`, trunk multiplier `0.3`, warmup 200 steps, then cosine decay to a 0.1 LR fraction. |
| `--wd`, `--clip`, `--seed` | Decay `0.1`, gradient clipping `1`, random seed `0`. |
| `--log-every`, `--eval-every`, `--eval-chunks` | Log every 100 steps; evaluate every 1000 and at final step; 240 held-out chunks per domain. Unlike `read`, this evaluation budget is in **chunks**. |
| `--grow-k`, `--grow-min-age`, `--grow-mem-frac`, `--pool-max` | Growth increment 8, prune survival 3000 steps, memory brake `0.80`, expert ceiling 512. `--grow-k 0` disables this growth/pruning path. |
| `--replay`, `--replay-burst` | Replay-selection probability 0 by default; 8 consecutive chunks per revisit. |
| `--revert-factor`, `--max-reverts` | Reload best saved weights and halve LR if validation exceeds best ×1.5; stop after 3 reverts. Factor 0 disables this guard. |

There is no `stream --save`: improved validation automatically writes the
best model. Other states need not be retained at exit. Logs append to
`<out>/history.jsonl` with start, step, validation, growth/prune, save and done
records. Its LR scheduler, checkpoint policy and gradient reach differ from
`read`. It offers no `--precision`, `--optimizer` or paging-residency CLI;
compute starts from the precision module's default BF16 where autocast is
supported. Every expert must fit resident alongside training overhead.

## Depth diagnostic: `ponder-probe`

```bash
.venv/bin/python train.py --device cpu ponder-probe \
  --ckpt runs/trainer-demo/weights --data data_math_char \
  --task add --n 20 --max-digits 8
```

Loads weights (or a legacy `.pt` checkpoint), generates seeded arithmetic
prompts with 1–8 digits, and prints mean/max halting steps at the final prompt
position. `--n` is prompts per digit count. `--data` selects tokenizer metadata,
not training examples; generate `data_math_char/meta.json` with the arithmetic
builder above first. Task names come from `corpora/arithmetic.py`'s `TASKS`.
No answers are scored and no optimizer steps run: increasing depth with problem
size is a diagnostic, not proof of arithmetic accuracy or reasoning capability.
Defaults are checkpoint `weights`, data `data_math_char`, task `add`, n=20,
max digits=8. Use the monitoring wrapper for a CUDA probe.

## Monitoring, comparisons and troubleshooting

The trainer does not sample board power itself. The `replication.gpu_monitor`
wrapper shown above records timestamped board W, GPU/memory utilization,
clocks, temperature and limiting masks, plus a whole-process energy summary.
It requires the optional `nvidia-ml-py` package in `.venv`. Whole-process energy
includes loading, evaluation and compilation; use explicit workload intervals
for training-only comparisons. Its samples are read-only and do not change GPU
clock/power settings. Low power alone does not establish unused compute.

For comparable runs, retain the source revision, full command/config, checkpoint
identity, data split/order, optimizer history policy, compute backend/precision,
context, depth policy, residency, warmup/calibration, throughput denominator,
peak allocated/reserved VRAM and average/peak power plus J/new token. `read`
re-forwards old context, so its reported new-character rate is not total
forwarded-token throughput. Keep evaluation/sample overhead separate when
comparing steady training rates.

Matching seeds alone does not guarantee exact GPU replay. The Muon research
comparison qualifies deterministic Torch execution and fixes the process-local
Sonic reduction configuration; see its protocol before comparing small quality
differences. These controls are implemented by the research runner, not a new
trainer CLI option.

The existing [Gefen learning screen](replication/GEFEN_PAGING.md),
[historical Muon variant comparison](replication/MUON_VARIANTS.md),
[ordinary FP8/Gefen experiment](replication/FP8_GEFEN.md),
[training profile measurements](replication/results/rtx5090_training_profile.json) and
[backend qualification](replication/EXPERT_BACKENDS.md) describe their tested
workloads and limits. They are experiments, not guarantees for a new model size.

| Symptom | Check |
| --- | --- |
| CUDA OOM | Active context, recurrent BPTT/depth, residency, expert checkpointing and competing desktop VRAM. RAM cache capacity is host memory, not a GPU cap. |
| “No readable text” | Paths, binary/hidden directory exclusions, UTF-8 validity. `read` skips `.bin`; use text or `stream` with packed data. |
| LR does not react to held-out | Enable `--sample-every` and provide a valid validation directory. `read --eval-every` is unused. |
| Shapes/settings unchanged after editing YAML | Existing checkpoint owns learned shapes. Create a separate fresh model to change width/block counts. |
| Missing/partial weights | A manifest alone contains no trained tensors; `read` creates a new model there. Partial foreign directories are rejected, not repaired automatically. |
| Gefen checkpoint refuses AdamW | Resume with Gefen; a deliberate optimizer-state reset/migration is separate work. Preserve the complete directory. |
| Gefen compiler preflight / Ninja failure | Keep `.venv/bin` on `PATH` and use the pinned optional setup/patch instructions; do not change system toolkits as a workaround. |
| Sonic unavailable or unsupported | Use the reference backend or the qualified source-patched stack; there is no silent fallback. |
| Experiments overwrite logs/plots | Give runs unique sample/history/trace paths. Plot image paths are shared; use `--no-plots` when isolating runs. |

# Gefen, Muon and master precision on the RTX 5090

Gefen is the strongest next **memory** candidate measured here: it retains FP32
parameters while reducing optimizer state from AdamW's 8 bytes/parameter to
about 1.01 on one real checkpoint gradient snapshot. This is research, not a
production optimizer replacement or evidence of equal learning quality.
The custom expert pager cannot yet preserve these optimizers' states.

## Local measurements

RTX 5090 32 GB, driver 595.91.07, Python 3.12.14, Torch 2.14.1+cu130.
All Python execution and dependencies use the repository `.venv`. No system
packages, drivers, clock settings or power limits were changed. Gefen 1.2.3
was built from `e735f851a7ebdf0106da87126458c86e3e583d99`.

The isolated optimizer benchmark has eight synthetic experts, hidden 512,
width 2048, three projections each: 25,165,824 FP32 parameters and FP32
gradients. Gradients have correlated row/column magnitudes and remain fixed.
`matrices` means 24 separate 2D parameters; `slabs` means three 3D banks.
Muon implementations reject 3D parameters and are measured only per matrix.
AdamW is Torch fused AdamW; Muon is Torch Muon; Gefen and GefenMuon use their
fused implementations. Momentum/betas, learning rate and other settings are
recorded in the probes. These are not tuned learning comparisons.

Each candidate has two runs in paired ABBA order, five warmups, 1.1 seconds
of settling, then a three-second measured interval. CUDA synchronization is
included in step latency. Compilation and first-step calibration are outside
timings but inside power monitoring. No forward/backward is timed here.

| Layout | Optimizer | Median step ms, range | State bytes/parameter | Average / peak sampled W | Board J/step | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| matrices | AdamW | 0.5034–0.5037 | 8.000 | 365 / 375 | 0.185 | 0.382 |
| matrices | Gefen | 0.4441–0.4450 | 1.012 | 482 / 490 | 0.215 | 0.218 |
| matrices | Muon | 5.3241–5.3245 | 4.000 | 357 / 363 | 1.904 | 0.321 |
| matrices | GefenMuon | 5.4247–5.4259 | 1.006 | 358 / 365 | 1.945 | 0.251 |
| slabs | AdamW | 0.4825–0.4836 | 8.000 | 375 / 381 | 0.182 | 0.430 |
| slabs | Gefen | 0.3091–0.3093 | 1.012 | 594 / 602 | 0.184 | 0.266 |

Gefen reduces measured optimizer latency by about 12% per matrix and 36% for
slabs. Its energy per step is about 16% higher per matrix and approximately
equal for slabs. Higher power alone is not an efficiency improvement. Sensor
averaging, desktop load and short windows limit power precision; the sampled
602 W peak is a sensor reading, not a changed 600 W power limit. Optimizer
steps/second are the reciprocal of latency; token throughput is undefined in
this optimizer-only workload. Raw data retain timestamps, utilization, clocks,
temperature and limiting masks, including compilation and warmup.

Stock Muon and GefenMuon take about 10–11 times AdamW's optimizer latency in
this layout. Their five-step Newton–Schulz work makes optimized independent
matrix batching worth testing. These timings cannot determine end-to-end
speed or time to a target validation loss.

### Real checkpoint state size

A read-only load of the downloaded checkpoint produced one Sonic BF16
training gradient on 4096 TinyStories tokens at depth 24, with gradient norm
clipping at 1. Gefen then stepped **detached copies** of the resident expert
weights and gradients. It was never attached to the live model or pager.

| Layout | Parameters | Optimizer state bytes | Bytes/parameter | First-step calibration s |
| --- | ---: | ---: | ---: | ---: |
| three slabs | 100,663,296 | 101,057,536 | 1.00392 | 3.663 |
| independent experts | 100,663,296 | 101,843,968 | 1.01173 | 3.494 |

This saves about 0.655 GiB versus FP32 AdamW moments on this bank. Compression
depends on gradients and chosen partition periods; it is not guaranteed to
be eightfold. A separate constant-gradient probe selected full second-moment
storage and used about 5.078 bytes/parameter for Gefen at the small probe shape.
One initial batch does not establish compression throughout training, learned
codebook stability, convergence, or performance with stochastic FP4 gradients.
The checkpoint manifest SHA-256 and resident expert identities are in the data.
No checkpoint was created or modified.

## FP32 masters versus BF16

The present reason for FP32 masters is retaining small updates. Around weight
0.013, adjacent BF16 values differ by about 6.10e-5; a 1.8e-5 update can round
away on every step. In a controlled constant-gradient test, FP32 parameters
changed while **all four tested optimizers left BF16 parameters unchanged**
over 33 updates. Gefen's compressed state does not add stochastic rounding to
parameter writeback. Muon's different update direction is not a precision fix.

Repeatedly paging an *unchanged* representable BF16 value through FP32 and back
is idempotent. The previous precision documentation implied cumulative damage
from this operation; that claim is corrected. Initial narrowing and rounding
each new optimizer update are the relevant losses. Live AdamW GPU moments are
FP32; this repo's disk moment storage already uses BF16. The documentation
correction changes no executable code, checked by AST comparison against dev.

Changing masters alone from FP32 to BF16 saves 2 bytes/parameter. With FP32
gradients, two FP32 moments and a BF16 compute cache, the current 18-byte floor
would become 16 bytes: about 11%, not half of training memory. Eliminating the
compute cache could save more but needs a separate design. Calling
`model.bfloat16()` also changes gradient accumulation and commonly optimizer
state dtypes, so it is not a controlled master-only experiment.

Compressing Adam state from 8 to approximately 1 byte while retaining FP32
masters saves about 7 bytes/parameter in the measured case. At the crossover
experiment's 805M-parameter resident bank this would save about 5.2 GiB if its
compression ratio holds, versus 1.5 GiB from masters alone; this is arithmetic,
not a measured full-model fit.

BF16 masters are possible with qualified stochastic rounding or compensated
updates. Our CPU rounding oracle preserves tiny updates in expectation; it is
not an implemented training optimizer. Production qualification must include
writeback, recurrent gradient accumulation, paging and checkpoint/RNG resume.
See [low precision training](https://arxiv.org/abs/2010.06192) and
[AdamW with stochastic rounding](https://github.com/amazon-science/adamw-sr).

## Integration and candidate hybrids

`PagedPool` recognizes `exp_avg` and `exp_avg_sq` and serializes per-expert
Adam m/v. The contract probe demonstrates that `_moment_state()` returns
`None` after Gefen has allocated state. Gefen needs encoded momentum,
magnitudes, partition metadata, step counters and its learned codebook.
Muon needs a momentum buffer; GefenMuon uses encoded momentum. Optimizer
`state_dict` round trips passed in isolation for all four optimizers, but the
application's custom paging path does not use that contract.

Before replacement, implement a versioned optimizer-state codec preserving
expert UID, per-expert geometry/partitions, optimizer-global state and steps.
Prove eviction A→B→A matches uninterrupted optimization; cover checkpoint
resume and pool growth. Define step-clock semantics explicitly: changing from
shared slabs to per-expert parameters can otherwise change the optimizer.

A hybrid is plausible: Muon-family updates for independent hidden expert
W1/W3/W2 and suitable attention/projection matrices, with AdamW or qualified
Gefen on embeddings/tied output heads, normalization, routers, gates, depth
embeddings and halting heads. Classify by module role and parameter identity,
not merely tensor rank. Never flatten the expert dimension into one Muon
matrix; batch independent 2D expert transforms. Learning rate/decay tuning and
matched multi-seed held-out loss runs are required, including native FP4.

Primary implementations and variants inspected:

- [Gefen](https://github.com/ndvbd/Gefen) and its
  [paper](https://arxiv.org/abs/2606.13894): measured above, including GefenMuon.
- [Muon](https://github.com/KellerJordan/Muon) and
  [Torch Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html):
  the Torch implementation was measured above.
- [NorMuon](https://github.com/zichongli5/NorMuon): neuron-wise adaptive scaling;
  a learning candidate, not assumed equivalent to ordinary Muon.
- [PolarExpress](https://github.com/NoahAmsel/PolarExpress): improved polynomial
  polar-factor approximations, relevant to orthogonalization cost.
- [Gram Newton–Schulz](https://github.com/Dao-AILab/gram-newton-schulz):
  smaller Gram-matrix iterations and hardware-aware implementation. Its README
  targets Hopper/B200/B300 and older dependency pins. Our installed QuACK
  0.6.5 symmetric GEMM explicitly includes SM120 dispatch, making a local
  compatibility probe worthwhile without downgrading the qualified stack.
- [MuonClip in Kimi K2](https://raw.githubusercontent.com/MoonshotAI/Kimi-K2/main/tech_report.pdf):
  attention QK stability work, requiring model-level integration.
- [Grouped Muon with stochastic rounding](https://huggingface.co/kernels/phanerozoic/muon):
  claims grouped kernels and BF16 writeback; source/architecture compatibility,
  arithmetic and learning behavior still need local qualification.

The variants in the last five bullets were **not executed** in this survey.
The next implementation priority is Gefen-aware paging, then a batched Muon
kernel comparison and hybrid learning tests. Neither the optimizer memory
result nor the FP4 layer crossover establishes model learning viability.

## Build qualification and evidence

Gefen's original preflight rejects toolkit 13.1 versus Torch CUDA 13.0.
`patches/gefen-cuda-minor-opt-in.patch` permits only an explicitly opted-in
same-major mismatch via `MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH=1`.
The default exact-version guard and major-version rejection remain tested.
Torch's installed extension builder itself accepts this minor mismatch with
a warning. The patch changes preflight only, not CUDA kernel arithmetic.
Existing toolkit 13.1.115/GCC 15.2 compiled the extension with `MAX_JOBS=2` and
`TORCH_CUDA_ARCH_LIST=12.0`. Initial failed preflight logs are retained.

Fused Gefen versus its unfused reference passed four same-state gradient
patterns with relative update errors 2.64e-7 to 8.26e-7 (limit 0.005).
Independent optimizer state-dict continuation was bit-identical at both
FP32/BF16 in the retention probe. The branch's full suite passed **81 tests**
under GPU power monitoring. These checks establish local arithmetic and
serialization behavior, not application paging or convergence.

Reproduce from this worktree using the main repository venv, after installing
the pinned optional Gefen source with the recorded preflight patch:

```sh
MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 \
/home/nir/ml-tools/mini-AGI/.venv/bin/python -m replication.probe_optimizer_candidates \
  --mode latency --optimizer gefen --layout matrices --dtype fp32 \
  --out /home/nir/ml-tools/mini-AGI/runs/performance/optimizers/reproduction.json
```

See [`results/optimizer_survey.json`](results/optimizer_survey.json) and
[`results/optimizer_survey_evidence.tar.gz`](results/optimizer_survey_evidence.tar.gz).
The archive contains 61 raw JSON/log files plus a verified SHA-256 manifest.
`summarize_optimizer_survey.py` assembles this local evidence without GPU work.
Probe sources, configuration and dependency provenance are retained. This
feature branch is research-only and unmerged pending production integration
and learning qualification.

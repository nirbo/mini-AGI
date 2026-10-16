# Replaceable expert computation and SonicMoE integration

`minagi.expert_compute.compute_experts` now owns expert arithmetic over accepted
routes. `PooledMLP` still owns routing, causal admission, capacity filtering,
gates and usage statistics. Pools still own parameters, paging, cast-cache
generations and optimizer moments. No checkpoint keys or formats change.

The default backend is **reference**. It retains exact-size and padded PyTorch
dispatch, deeper residual experts, existing precisions and the small-inference
path. The optional **sonic** backend uses SonicMoE's externally routed BF16
SwiGLU forward/backward. It does not call SonicMoE's router.

## Selection and constraints

Set `MINAGI_EXPERT_BACKEND=sonic` for a training/evaluation/generation process;
unset it or use `reference` for the existing implementation. A call site's
`expert_backend` attribute can override this process setting. Backend selection,
compute dtype, routes and weight generations are bound before checkpoint replay.
`MINAGI_DISPATCH=exact|padded` applies to the reference backend only.

Sonic requires CUDA BF16 compute, depth-1 experts and hidden/expert widths
divisible by 16. The adapter accepts QuACK's SM90, SM100 and SM120 GPU families;
these measurements qualify the RTX 5090 only. Tested dimensions include
hidden/expert widths 32/64, 64/128 and 512/2048. Unsupported settings and missing
optional packages raise an error rather than silently changing backends.
CPU/FP32/deeper experts remain available through `reference`.

Dependencies remain optional and lazy-imported. Use the source revisions and
venv-only installation directions in
[`NVFP4_FEASIBILITY.md`](NVFP4_FEASIBILITY.md) and
[`results/nvfp4_environment.json`](results/nvfp4_environment.json).
Apply [`patches/sonicmoe-fixed-gemm-config.patch`](patches/sonicmoe-fixed-gemm-config.patch)
to the exact pinned SonicMoE source with `git -C <sonic-path> apply
--unidiff-zero <absolute-patch-path>` and reinstall that source with
`uv pip install --python .venv/bin/python --no-deps --reinstall <sonic-path>`.
The adapter checks the patch marker before executing Sonic. The patch passes
`tuned=False` to six public QuACK GEMM calls, using the library's architecture
default configuration. It changes no routing or activation formula. A regression
guards against entering the online GEMM autotuner during forward or backward.
No other dependency or system changes were made for this integration.

The stock tuning key groups leading dimensions but retains exact token counts
in transposed weight-gradient inputs. Adaptive halting consequently triggers
repeated tuning: the first unpatched full-model update took 191.9 seconds.
That run was stopped before measurement and preserved as a pilot. Extending
the token-count cache key was rejected after another sparse backward memory
error. The qualified configuration avoids online GEMM tuning entirely; offline
configuration tuning is future work. Successful fixed-shape microbenchmarks
with stock Sonic should not be mistaken for this model's deployment behavior.

The interface receives token-major assignments *after* capacity filtering,
with physical slot IDs and scores including the expert gates. Scores of zero
remain in the graph because their derivatives can be nonzero. Completely empty
assignment sets return zeros, matching the reference. Sonic appends one constant
zero input row and one zero-score dummy assignment per resident expert. It
discards that row on return. Every native backward
reduction group is nonempty, while expert count stays fixed across calls. The
dummy contributes zero weight/input gradients and no routing statistics. This
also avoids host synchronization and repeated metadata JIT for varying numbers
of compacted experts. Real zero-score assignments keep their derivatives.

Gate/up packing is per use and inside the checkpointed region. There is no
shared differentiable packed-weight cache. Existing detached BF16 storage and
separate per-use FP32 gradient edges are retained. Eviction, optimizer updates,
parameter replacement and device/dtype changes retain their existing cache
invalidation rules. Further packed-data caching is a separate optimization.

## Verification

The previous default dispatch from `d0f581a` and the refactored reference were
run independently on the downloaded real checkpoint. Six deterministic updates
produced identical losses, parameter hashes, clipped-gradient hashes, optimizer
state hashes and slot identities, separately with checkpointing on and off.
Hashing runs are correctness checks, not performance measurements.

New tests cover independent FP64 assignment-by-assignment output/gradient
oracles for depth-1 and depth-2 reference experts, exact/padded dispatch,
checkpoint replay, missing token routes, unused experts and zero-score
derivatives. CUDA Sonic tests check all input/weight/score derivatives against
the fixed-route BF16 reference (finite and relative L2 < 0.025), including
missing token routes, negative/zero scores and unused expert gradients.

Model checks exercise recurrence, adaptive halting, causal capacity, stochastic
admission, independent sequences, multiple frames before one backward, cached
and uncached cast generations, and checkpoint recomputation. Paging tests load
experts with distinct Adam moments, compute Sonic gradients, and verify the
resulting moment values after eviction and reload.

Fusion changes BF16 rounding. During integration, a toy comparison changed one
near-tied route after recurrence: whole-model per-parameter gradient equality
across backends is therefore not a valid requirement for that example. One
small depth-embedding gradient differed by 10.6% relative L2 (reference norm
2.03e-6, maximum absolute discrepancy 7.15e-8). Fixed-route derivative checks
remain per tensor; the full-model comparison bounds total gradient relative L2
by 0.05 and logit relative L2 by 0.02. Within Sonic, cast caching and checkpoint
replay must retain the same routes/outputs and match all parameter gradients
within rtol 1e-5, atol 1e-9. This does not establish long-run learning quality.

The initial append-prefix test found a one-ULP BF16 logit difference in one
element. Diagnostic hooks showed **bit-identical expert outputs and pre-head
states** at every recurrent pass; the output-head cuBLAS GEMM changed rounding
between row counts 5 and 24. The expert-causality regression uses a fixed
one-row head projection, retaining BF16 arithmetic, to isolate the backend.
It does not claim bit-identical production logits across all CUDA matrix
shapes. The existing default-backend causality tests remain unchanged.

An initial cold-autotuning test with unused expert groups hit an illegal memory
access in native backward. Warm-cache repeats alone were insufficient evidence:
forced retuning reproduced the failure. Compacting unused groups made that
forced-retuning check pass, but changing compacted expert counts caused
repeated metadata compilation in the real model. The final adapter instead
uses neutral dummy padding and the fixed default GEMM configuration. It never
submits empty expert reduction groups. This experiment
does not identify every affected upstream kernel configuration.

## Full-model measurements

**Two runs per backend/configuration; measured update windows:**

| Expert checkpointing | Backend | Median s/update | Window new tokens/s | Average / peak W | J/new token | Peak GiB, including warmup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| On | reference | 0.376 | 5367–5387 | 270–273 / 297–298 | 0.0501–0.0510 | 5.26–5.33 |
| On | sonic | 0.296–0.300 | 6270–6768 | 261–271 / 295–316 | 0.0401–0.0417 | 4.25–4.26 |
| Off | reference | 0.319–0.322 | 6144–6247 | 270–271 / 295–303 | 0.0433–0.0439 | 13.24–13.62 |
| Off | sonic | 0.271–0.274 | 7220–7236 | 254 / 271–281 | 0.0351–0.0352 | 11.01–11.33 |

Aggregate measured-window throughput improved **21.1% with checkpointing** and
**16.7% without**. Board energy per new token fell **19.1%** and **19.5%**,
respectively. These use total tokens/window duration and integrated energy,
rather than inferring speed from the median step alone.

All **77 tests passed** on the final implementation. Every final timing run completed 40 finite-loss/finite-gradient updates including warmup. The selected timings have no multi-second JIT outliers; per-update maxima and all raw timing arrays are preserved. No backend was enabled by default.

Machine-readable results and source/environment hashes: [`results/expert_backend_verification.json`](results/expert_backend_verification.json). Raw profiles, telemetry, test/diagnostic logs, reference-comparison scripts and SHA256 manifest: [`results/expert_backend_evidence.tar.gz`](results/expert_backend_evidence.tar.gz). Rejected tuning and compaction pilots are archived separately inside that bundle and excluded from these measurements.

The profiler opens the checkpoint read-only and uses real text, BF16 arithmetic,
FP32 masters, context 4096, stride 2048, resident capacity 32, a fixed 24-pass
ceiling, fresh fused AdamW, gradient clipping and the existing weight cache.
Each run starts from the same weights/seed, warms up for 8 updates and measures
32 updates. Backend order is reference/Sonic/Sonic/reference in each checkpoint
configuration. Optimizer updates remain in memory; no trained artifact is saved.

Precision differences can change routing and the subsequent learning trajectory;
these are end-to-end timing observations, not identical-work FLOP comparisons or
convergence results. Loading and the eight warmup updates, including their JIT
work, are excluded from measured windows, as are saves, evaluation and sampling.
Any remaining compilation during measured updates is counted. Initial JIT on a
fresh installation may take minutes.
Peak allocated VRAM includes warmup/JIT allocations as well as training; it
is not an isolated persistent-training-memory measurement. The small numerical
difference and short finite-loss runs do not establish
matched held-out quality. Sonic remains opt-in.

Power is sampled read-only via NVML every 100 ms. Energy summaries align to the
profiler's measured update window, excluding loading and warmup. NVML may
internally average power; sampled peaks are not instantaneous maxima. Board
energy includes desktop activity, excludes CPU/system energy, and does not
measure Tensor Core occupancy. Raw samples include clocks, utilization,
temperature and clock-limiting flags. No power/clock settings were changed.
The checkpointed Sonic runs include transient software power-capping flags even
though average power was 261–271 W; averaged watts alone do not establish
unused compute capacity.

Example measurement, using the repository venv:

```bash
.venv/bin/python -m replication.gpu_monitor \
  --out runs/performance/sonic.power.json -- \
  .venv/bin/python replication/profile_training.py \
  --weights /mnt/fast/models/mini-AGI/undertrained/weights \
  --text /mnt/fast/models/mini-AGI/evaluation-data/TinyStoriesV2-GPT4-valid.txt \
  --expert-backend sonic --warmup 8 --steps 32 --no-profile \
  --out runs/performance/sonic.json
```

Use `--expert-backend reference` for the control and `--no-checkpoint` for the
other memory/compute tradeoff. The monitor's CLI prints whole-process power;
use `timing_interval_unix` and the raw samples for workload-only summaries.
Run GPU commands sequentially, with no competing GPU jobs.

```bash
.venv/bin/python -m replication.gpu_monitor \
  --out runs/performance/backend-tests.power.json -- \
  .venv/bin/python -m unittest discover -s tests -v
```

Native NVFP4 remains an independent future backend. This BF16 adapter does not
implement FP4 grouped backward, causal activation scaling, QAT, or a low-precision
checkpoint format. Its contract leaves routing and pool ownership outside the
backend so that a future NVFP4 implementation can reuse those guarantees.

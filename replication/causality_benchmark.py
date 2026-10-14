"""CUDA speed check with random weights; no data/ or weights/ files are used.

Compare code revisions with --root, keeping this benchmark script identical.
Run: .venv/bin/python replication/causality_benchmark.py --out /tmp/speed.json
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
ap.add_argument("--out")
ap.add_argument("--depth", type=int, default=13)
args = ap.parse_args()
sys.path.insert(0, args.root)

import torch
from minagi.recur import RecurConfig, RecurCoder
from minagi.pool import PooledMLP
from minagi.paged import PagedPool
from minagi.precision import set_compute_dtype, amp

if not torch.cuda.is_available():
    raise SystemExit("This benchmark requires a CUDA GPU.")
if args.depth < 1:
    raise SystemExit("--depth must be positive")
torch.set_num_threads(4)
torch.manual_seed(42)
set_compute_dtype("bf16")
cfg = RecurConfig(vocab_size=265, d_model=512, n_head=8, d_ff=1408,
                  n_prelude=2, n_recur=1, n_coda=0, max_steps=args.depth, block=4096,
                  train_steps_mean=0, bptt_window=args.depth, halt_freeze=True,
                  halt_prior=0.072, use_pool=True, pool_experts=64, pool_max=64,
                  pool_d_ff=2048, pool_top_k=8, pool_capacity_factor=0)
model = RecurCoder(cfg)
weights = [{k: getattr(e, k).weight.detach().clone() for k in ("w1", "w3", "w2")}
           for e in model.pool.experts]
with tempfile.TemporaryDirectory(prefix="minagi-speed-") as tmp:
    pool = PagedPool(tmp, 512, 2048, 64, resident=32, ram_capacity=64,
                     device="cuda", read_only=True).cuda()
    for i, entry in enumerate(weights):
        pool.tiers.put(i, entry, dirty=False)
    del weights
    for site in model.modules():
        if isinstance(site, PooledMLP):
            site._pool[0] = pool
    model.pool = pool
    pool.attach_sites(model)
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pool.attach_optimiser(opt)
    text = torch.randint(0, 265, (1, 4096), device="cuda")
    target = torch.roll(text, -1, 1)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(8):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with amp("cuda"):
            _, loss = model(text, target)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert torch.isfinite(loss)
        if i >= 2:
            times.append(elapsed)
    peak = torch.cuda.max_memory_allocated() / 2**30
    model.eval()
    writing = []
    with torch.no_grad():
        for i in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.generate(text[:, :256], 16, adapt_strength=0)
            torch.cuda.synchronize()
            if i:
                writing.append(time.perf_counter() - start)
    report = {
        "root": args.root, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "precision": "bf16",
        "width": 512, "expert_width": 2048, "experts": 64, "resident": 32,
        "top_k": 8, "context": 4096, "fixed_depth": args.depth, "capacity_factor": 0,
        "training_step_seconds": times,
        "training_median_seconds": statistics.median(times),
        "training_chars_per_second": 4096 / statistics.median(times),
        "peak_allocated_gib": peak,
        "generation_256_prompt_16_chars_seconds": writing,
        "note": ("Random weights and bytes, warm RAM cache; timing includes backward "
                 "and Adam. Generation timing includes prompt prefill. "
                 "No quality claim or disk-I/O benchmark."),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

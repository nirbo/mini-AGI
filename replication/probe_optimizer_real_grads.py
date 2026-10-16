"""Measure Gefen state compression on one real checkpoint gradient batch.

The model is read-only on disk. Only detached parameter copies are optimized;
the application pager never receives unsupported optimizer state.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from gefen import Gefen
from minagi.ingest import as_stream
from minagi.pool import PooledMLP
from minagi.precision import amp, set_compute_dtype
from minagi.recur import load_any
from replication.gpu_monitor import PowerMonitor
from replication.probe_optimizer_candidates import state_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": ["Single first-batch gradient snapshot, Sonic BF16, fixed depth 24 and context 4096, no parameter learning comparison.",
                         "Gefen steps detached copies, not the live paged model. Calibration/state sizes may change with data, gradients and architecture."]}
    with PowerMonitor() as monitor:
        try:
            torch.manual_seed(42)
            torch.set_num_threads(4)
            torch.cuda.set_per_process_memory_fraction(.75)
            set_compute_dtype("bf16")
            model, _ = load_any(str(args.weights), torch.device("cuda"), read_only=True)
            model.cfg.max_steps = model.cfg.bptt_window = 24
            model.cfg.train_steps_mean = 0
            for site in model.modules():
                if isinstance(site, PooledMLP):
                    site.expert_backend = "sonic"
                    site.grad_checkpoint = True
            model.train()
            data = as_stream(str(args.text))[:4097].astype(np.int64)
            ids = torch.tensor(data, device="cuda")[None]
            with amp("cuda"):
                _, loss = model(ids[:, :-1], ids[:, 1:])
                loss = loss + model.cfg.pool_aux * model.pool_aux() + model.pool_balance()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            values = [getattr(model.pool, name).detach().clone() for name in ("w1", "w3", "w2")]
            grads = [getattr(model.pool, name).grad.detach().clone() for name in ("w1", "w3", "w2")]
            report.update(loss=float(loss.detach()), resident_expert_ids=list(model.pool.slots),
                          manifest_sha256=hashlib.sha256((args.weights / "manifest.json").read_bytes()).hexdigest(),
                          gradient_nonzero_fractions=[float(g.count_nonzero() / g.numel()) for g in grads])
            del model, loss, ids
            gc.collect()
            cases = []
            for layout in ("slabs", "matrices"):
                ws = values if layout == "slabs" else [w for slab in values for w in slab.unbind()]
                gs = grads if layout == "slabs" else [g for slab in grads for g in slab.unbind()]
                parameters = [torch.nn.Parameter(w.clone()) for w in ws]
                opt = Gefen(parameters, lr=1e-5, betas=(.9, .95), weight_decay=.1, fused=True)
                for p, g in zip(parameters, gs):
                    p.grad = g
                start = time.perf_counter()
                opt.step()
                torch.cuda.synchronize()
                if not all(bool(torch.isfinite(p).all()) for p in parameters):
                    raise AssertionError("non-finite calibrated parameter")
                count = sum(p.numel() for p in parameters)
                cases.append({"layout": layout, "parameter_count": count, "state_bytes": state_bytes(opt),
                              "state_bytes_per_parameter": state_bytes(opt) / count,
                              "first_step_calibration_seconds": time.perf_counter() - start,
                              "periods": [s["automatic_period"] for s in opt.state.values()]})
                del opt, parameters
                gc.collect()
            report.update(status="passed", cases=cases)
        except BaseException as exc:
            report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            monitor.sample()
            monitor.save(args.out.with_suffix(".power.json"))
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

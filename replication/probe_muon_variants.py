"""Isolated matched 8-expert optimizer benchmark; no model files or training."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from replication.compare_gefen_training import tensor_bytes
from replication.gpu_monitor import PowerMonitor, summarize_samples
from replication.muon_candidates import VARIANTS, SOURCES, matrix_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimizer', choices=('adamw', 'gefen', *VARIANTS), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=5.)
    args = parser.parse_args()
    report = {'optimizer':args.optimizer, 'sources':SOURCES, 'dtype':'FP32 masters/gradients',
              'geometry':{'experts':8, 'hidden':512, 'width':2048},
              'limit':'Optimizer-only fixed correlated gradients, not learning or a full trainer speedup.'}
    with PowerMonitor() as monitor:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        shapes = [(2048,512), (2048,512), (512,2048)] * 8
        ps = [torch.nn.Parameter(torch.randn(s, device='cuda')*.02) for s in shapes]
        for p in ps:
            p.grad = (torch.randn(p.shape[0],1,device='cuda')*torch.randn(1,p.shape[1],device='cuda')
                      + .1*torch.randn_like(p))*.001
        if args.optimizer == 'adamw':
            opt = torch.optim.AdamW(ps, lr=1e-4, betas=(.9,.95), weight_decay=.1, fused=True)
        elif args.optimizer == 'gefen':
            from gefen import Gefen
            opt = Gefen(ps, lr=1e-4, betas=(.9,.95), weight_decay=.1, fused=True)
        else:
            opt = matrix_optimizer(args.optimizer, ps, lr=1e-4)
        t0=time.perf_counter()
        for _ in range(3):
            opt.step()
        torch.cuda.synchronize()
        report['startup_seconds']=time.perf_counter()-t0
        settle=time.perf_counter()
        while time.perf_counter()-settle < 1.:
            opt.step()
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        times=[]
        while time.perf_counter()-start < args.seconds:
            t0=time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-t0)
        end=time.perf_counter()
        report.update(start_monotonic=start,end_monotonic=end,iterations=len(times),
                      median_step_seconds=statistics.median(times),
                      state_bytes=tensor_bytes(dict(opt.state)),
                      state_bytes_per_parameter=tensor_bytes(dict(opt.state))/sum(p.numel() for p in ps),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      finite=all(bool(torch.isfinite(p).all()) for p in ps))
    report['power']=summarize_samples(monitor.samples,start,end)
    report['estimated_board_joules_per_step']=report['power']['estimated_board_energy_j']/len(times)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    monitor.save(args.out.with_suffix('.power.json'))
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['finite']:
        raise AssertionError('non-finite optimizer benchmark weights')


if __name__ == '__main__':
    main()

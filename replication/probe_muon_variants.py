"""Isolated matched 8-expert optimizer benchmark; no model files or training."""

import argparse
import json
import os
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
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--deterministic',action='store_true')
    args = parser.parse_args()
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
    report = {'optimizer':args.optimizer, 'sources':SOURCES, 'dtype':'FP32 masters/gradients',
              'deterministic_algorithms':args.deterministic,
              'geometry':{'experts':args.experts, 'hidden':args.hidden, 'width':args.width},
              'limit':'Optimizer-only fixed correlated gradients, not learning or a full trainer speedup.'}
    with PowerMonitor() as monitor:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.backends.cuda.matmul.allow_tf32 = False
        shapes = [(args.width,args.hidden), (args.width,args.hidden), (args.hidden,args.width)] * args.experts
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

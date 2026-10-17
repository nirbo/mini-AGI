"""Attribute FP8 quantization/rescaling/GEMM work; run externally monitored.

Profiler overhead makes these diagnostics unsuitable as throughput benchmarks.
FP32 gradients are retained; no optimizer or checkpoint is modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.profiler import profile,ProfilerActivity,record_function

from replication import fp8_experts as fp8


def trace_kernel_summary(path):
    """Count actual GPU work; exclude duplicate GPU annotation ranges."""
    groups = {}
    for event in json.loads(path.read_text())['traceEvents']:
        category = event.get('cat')
        if event.get('ph') != 'X' or category not in ('kernel', 'gpu_memcpy'):
            continue
        name = event['name']
        if category == 'gpu_memcpy':
            group = 'copy'
        elif 'quackgemm' in name:
            group = 'gemm'
        elif 'abs_amax' in name:
            group = 'quantize'
        elif 'mul_transpose' in name:
            group = 'rescale'
        else:
            group = 'other'
        row = groups.setdefault(group, dict(count=0, total_us=0., names=[]))
        row['count'] += 1
        row['total_us'] += event['dur']
        if name not in row['names']:
            row['names'].append(name)
    return groups


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.manual_seed(42)
    report=dict(cases=[],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__),Path(fp8.__file__))},
                limits=['Profiler overhead; kernel/CPU attribution, not throughput measurements.',
                        'Nested inclusive device times overlap and must not be summed.',
                        'Native GEMM scope includes allocation and the nested output-rescale operation.',
                        'CPU and GPU annotations can share a name; use device_type to disambiguate.',
                        'Kernel groups sum actual kernel/copy durations, excluding annotation ranges; not wall time.'])
    for m,k,n in ((1024,512,2048),(4096,4096,4096)):
        for cached in (False,True):
            x=torch.randn(m,k,device='cuda',requires_grad=True)
            w=(torch.randn(n,k,device='cuda')/k**.5).requires_grad_()
            g=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
            cache=fp8.WeightCache()
            def step():
                x.grad=w.grad=None
                if not cached:cache.clear()
                with record_function('fp8::bind_weight_cache'):
                    packed=cache.bind((w,))[0]
                fp8.linear(x,w,packed).backward(g)
            for _ in range(4):step()
            torch.cuda.synchronize()
            quant,mm,scale=fp8.quantize_rows,fp8.native_mm,fp8.rescale
            def quantize(*a,**kw):
                with record_function('fp8::quantize_rows'):return quant(*a,**kw)
            def native(*a,**kw):
                with record_function('fp8::native_mm_plus_rescale'):return mm(*a,**kw)
            def rescale(*a,**kw):
                with record_function('fp8::rescale_output'):return scale(*a,**kw)
            with patch.object(fp8,'quantize_rows',quantize),patch.object(fp8,'native_mm',native), \
                 patch.object(fp8,'rescale',rescale):
                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
                    for _ in range(16):step()
                    torch.cuda.synchronize()
            tag=f'{m}-{k}-{n}-{"cached" if cached else "uncached"}'
            trace=args.out.with_name(args.out.stem+'-'+tag+'.trace.json')
            prof.export_chrome_trace(str(trace))
            rows=[dict(name=e.key,device_type=str(e.device_type),count=e.count,cpu_total_us=e.cpu_time_total,
                       self_cpu_us=e.self_cpu_time_total,device_total_us=e.device_time_total,
                       self_device_us=e.self_device_time_total) for e in prof.key_averages()]
            report['cases'].append(dict(shape_mkn=[m,k,n],cached=cached,iterations=16,trace=str(trace),
                                        gradient_dtypes=[str(x.grad.dtype),str(w.grad.dtype)],operators=rows,
                                        kernel_groups=trace_kernel_summary(trace)))
            args.out.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(shape=[m,k,n],cached=cached,
                                  stages=[e for e in rows if e['name'].startswith('fp8::')])),flush=True)
            del x,w,g,cache,prof
            torch.cuda.empty_cache()
    report['status']='passed'
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

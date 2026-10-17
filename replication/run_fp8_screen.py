"""Serial ordinary-FP8/Gefen experiments with external NVML telemetry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from replication.gpu_monitor import PowerMonitor,summarize_samples

ROOT=Path(__file__).resolve().parents[1]


def validate(report):
    if report['status']!='passed':raise AssertionError('failed FP8 trial')
    for name,digest in report['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise AssertionError(f'measured source changed: {name}')


def monitored(command,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with PowerMonitor() as monitor:
        with path.with_suffix('.log').open('w') as log:
            code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    monitor.save(path.with_suffix('.external-power.json'),command=command,returncode=code)
    if code:raise RuntimeError(f'GPU command failed; see {path.with_suffix(".log")}')
    return monitor.samples


def learning(root,backend,seed,steps,fingerprint=False,cache=True):
    tag=f'{backend}-seed{seed}-steps{steps}'+('' if cache else '-uncached')
    path=root/(tag+'.json')
    if path.exists():
        r=json.loads(path.read_text());validate(r)
        cfg=r['configuration']
        if (cfg['backend'],cfg['seed'],cfg['steps'],cfg['fingerprint'],cfg['no_cache'])!=(backend,seed,steps,fingerprint,not cache):
            raise AssertionError('cached trial differs from requested configuration')
        if 'external_training_power' not in r:raise AssertionError('incomplete trial telemetry')
        return r
    command=[sys.executable,'-m','replication.compare_fp8_training','--backend',backend,
             '--seed',str(seed),'--steps',str(steps),'--out',str(path)]
    if fingerprint:command.append('--fingerprint')
    if not cache:command.append('--no-cache')
    print(json.dumps(dict(starting=tag)),flush=True)
    samples=monitored(command,path)
    r=json.loads(path.read_text());validate(r)
    r['external_training_power']=summarize_samples(samples,r['training_start'],r['training_end'],steps*512)
    r['external_measured_training_power']=summarize_samples(samples,r['measurement_start'],r['training_end'],
                                                            (steps-r['measurement_warmup_updates'])*512)
    path.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(finished=tag,test_ce=r['final_test']['mean_ce'],
                          seconds=r['training_end']-r['training_start'],
                          watts=r['external_training_power']['average_board_power_w'])),flush=True)
    return r


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('runs/fp8-screen'))
    ap.add_argument('--phase',choices=('repro','learning','probes'),required=True)
    args=ap.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    os.environ.update(PATH=str(ROOT/'.venv/bin')+os.pathsep+os.environ['PATH'],
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH='1',
                      TORCHINDUCTOR_COMPILE_THREADS='2',MAX_JOBS='2')
    if args.phase=='repro':
        all_runs=[]
        for backend in ('sonic','fp8'):
            a=learning(args.root/'repro-a',backend,42,8,True)
            b=learning(args.root/'repro-b',backend,42,8,True)
            for key in ('initial_model_sha256','first_gradient_sha256','final_model_sha256',
                        'final_optimizer_sha256','final_host_sha256','initial_tune','initial_test',
                        'final_tune','final_test','final_slots','training_window_indices'):
                if a[key]!=b[key]:raise AssertionError(f'replay mismatch: {backend} {key}')
            for left,right in zip(a['trace'],b['trace']):
                for key in ('objective','gradient_norm'):
                    if left[key]!=right[key]:raise AssertionError(f'trace replay mismatch: {backend} {key}')
            all_runs.append(a)
        if len({r['initial_model_sha256'] for r in all_runs})!=1:raise AssertionError('different initial weights')
        # Cached and uncached quantization must preserve the actual trajectory.
        uncached=learning(args.root/'repro-uncached','fp8',42,8,True,False)
        reference=all_runs[1]
        for key in ('final_model_sha256','first_gradient_sha256','final_optimizer_sha256',
                    'final_host_sha256','final_tune','final_test','final_slots'):
            if uncached[key]!=reference[key]:raise AssertionError(f'cache changes learning: {key}')
        (args.root/'repro-passed.json').write_text(json.dumps(dict(status='passed',
                    independent_processes_per_backend=2,identical_initial_weights=True,
                    fp8_cached_uncached_exact=True,steps=8),indent=2)+'\n')
    elif args.phase=='learning':
        if not (args.root/'repro-passed.json').exists():raise RuntimeError('run the replay gate first')
        for seed in (42,43,44):
            for backend in ('sonic','fp8'):
                r=learning(args.root,backend,seed,256)
                if seed==42:
                    oracle=json.loads((args.root/'repro-a'/f'{backend}-seed42-steps8.json').read_text())
                    validate(oracle)
                    if r['initial_model_sha256']!=oracle['initial_model_sha256']:raise AssertionError('changed initial weights')
                    for a,b in zip(r['trace'][:8],oracle['trace']):
                        if any(a[k]!=b[k] for k in ('objective','gradient_norm')):
                            raise AssertionError('256-update prefix differs from replay gate')
    else:
        for repeat in (1,2):
            for phase,large in (('linear',False),('linear',True),('experts',False),('experts',True)):
                path=args.root/f'{phase}-{"large" if large else "current"}-repeat{repeat}.json'
                if path.exists():raise RuntimeError(f'probe result already exists: {path}')
                command=[sys.executable,'-m','replication.probe_fp8','--phase',phase,'--seconds','3','--out',str(path)]
                if large:command.append('--large')
                if repeat==2:command.append('--reverse')
                print(json.dumps(dict(starting_probe=path.name)),flush=True)
                samples=monitored(command,path)
                r=json.loads(path.read_text())
                validate(r)
                for case in r['cases']:
                    b=case['benchmark']
                    case['external_power']=summarize_samples(samples,b['start_monotonic'],b['end_monotonic'],b['token_rows'])
                path.write_text(json.dumps(r,indent=2)+'\n')
                print(json.dumps(dict(finished_probe=path.name)),flush=True)


if __name__=='__main__':main()

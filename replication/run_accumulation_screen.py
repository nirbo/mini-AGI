"""Serial Gefen accumulator study with external NVML monitoring."""

import argparse
import json
import os
from pathlib import Path
import sys

from replication.run_fp8_screen import monitored, validate
from replication.gpu_monitor import summarize_samples

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('sonic','bf16-fp32','fp16-fp32','fp16-fp16','fp16-split4')


def learning(root, backend, seed, steps, fingerprint=False, cache=True):
    tag=f'{backend}-seed{seed}-steps{steps}'+('' if cache else '-uncached')
    path=root/(tag+'.json')
    if path.exists():
        r=json.loads(path.read_text());validate(r)
        cfg=r['configuration']
        assert (cfg['backend'],cfg['seed'],cfg['steps'],cfg['fingerprint'],cfg['no_cache']) == (backend,seed,steps,fingerprint,not cache)
        assert 'external_training_power' in r
        return r
    command=[sys.executable,'-m','replication.compare_accum_training','--backend',backend,
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
    ap.add_argument('--phase',choices=('repro','probes','learning'),required=True)
    ap.add_argument('--root',type=Path,default=Path('runs/accum-screen'))
    args=ap.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    os.environ.update(PATH=str(ROOT/'.venv/bin')+os.pathsep+os.environ['PATH'],
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH='1',
                      TORCHINDUCTOR_COMPILE_THREADS='2',MAX_JOBS='2')
    if args.phase=='repro':
        runs=[]
        for backend in BACKENDS:
            a=learning(args.root/'repro-a',backend,42,8,True)
            b=learning(args.root/'repro-b',backend,42,8,True)
            for key in ('initial_model_sha256','first_gradient_sha256','final_model_sha256',
                        'final_optimizer_sha256','final_host_sha256','initial_tune','initial_test',
                        'final_tune','final_test','final_slots','training_window_indices'):
                assert a[key]==b[key],(backend,key)
            assert [(x['objective'],x['gradient_norm']) for x in a['trace']] == [(x['objective'],x['gradient_norm']) for x in b['trace']]
            runs.append(a)
        assert len({r['initial_model_sha256'] for r in runs})==1
        # The qualified cache/replay machinery is shared; verify the most
        # modified split-K path against freshly converted weights as well.
        fresh=learning(args.root/'repro-uncached','fp16-split4',42,8,True,False)
        for key in ('first_gradient_sha256','final_model_sha256','final_optimizer_sha256',
                    'final_host_sha256','final_tune','final_test','final_slots'):
            assert fresh[key]==runs[-1][key],('cache',key)
        (args.root/'repro-passed.json').write_text(json.dumps(dict(status='passed',backends=BACKENDS,
                    independent_processes_per_backend=2,steps=8,identical_initial_weights=True,
                    split4_cached_uncached_exact=True),indent=2)+'\n')
    elif args.phase=='probes':
        for repeat in (1,2):
            for phase in ('gemm','linear','experts'):
                for large in (False,True):
                    path=args.root/f'{phase}-{"large" if large else "current"}-repeat{repeat}.json'
                    if path.exists():raise RuntimeError(f'probe result already exists: {path}')
                    command=[sys.executable,'-m','replication.probe_accumulation','--phase',phase,
                             '--seconds','3','--out',str(path)]
                    if large:command.append('--large')
                    if repeat==2:command.append('--reverse')
                    print(json.dumps(dict(starting_probe=path.name)),flush=True)
                    samples=monitored(command,path)
                    r=json.loads(path.read_text());validate(r)
                    for c in r['cases']:
                        b=c['benchmark']
                        c['external_power']=summarize_samples(samples,b['start_monotonic'],b['end_monotonic'],b['token_rows'])
                    path.write_text(json.dumps(r,indent=2)+'\n')
                    print(json.dumps(dict(finished_probe=path.name)),flush=True)
    else:
        gate=json.loads((args.root/'repro-passed.json').read_text());assert gate['status']=='passed'
        for seed in (42,43,44):
            for backend in BACKENDS:
                r=learning(args.root,backend,seed,256)
                if seed==42:
                    oracle=json.loads((args.root/'repro-a'/f'{backend}-seed42-steps8.json').read_text());validate(oracle)
                    assert r['initial_model_sha256']==oracle['initial_model_sha256']
                    assert [(x['objective'],x['gradient_norm']) for x in r['trace'][:8]] == [(x['objective'],x['gradient_norm']) for x in oracle['trace']]


if __name__=='__main__':main()

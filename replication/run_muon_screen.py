"""Serial GPU screen with external telemetry; never writes model checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from replication.gpu_monitor import PowerMonitor, summarize_samples
from replication.muon_candidates import VARIANTS, ROOT


def validate_learning_protocol(report, fingerprint):
    if not report.get('deterministic_algorithms_enabled') or report['configuration'].get('fingerprint')!=fingerprint:
        raise RuntimeError('incompatible deterministic execution protocol')
    if report.get('fixed_sonic_reduction')!=dict(kwargs={'BLOCK_H':256,'BLOCK_K':32},num_warps=4,num_stages=4):
        raise RuntimeError('Sonic reduction is not the qualified fixed configuration')
    for source,digest in report['sources'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'result source differs from current file: {source}')
    if fingerprint or report['configuration']['seed']!=42:
        return
    name=report['configuration']['optimizer']
    reference=ROOT/'runs/muon-deterministic-repro-fixed/a'/f'{name}-lr3e-05-seed42-steps8.json'
    if not reference.exists():
        raise RuntimeError('run the independent-process reproducibility phase first')
    earlier=json.loads(reference.read_text())
    count=8 if report['configuration']['lr']==3e-5 else 1
    if any(len(r[key]) < count for r in (report,earlier)
           for key in ('trace','training_window_indices')):
        raise AssertionError('incomplete training prefix for reproducibility check')
    if report['training_window_indices'][:count]!=earlier['training_window_indices'][:count]:
        raise AssertionError('data prefix differs from reproducibility oracle')
    for left,right in zip(report['trace'][:count],earlier['trace'][:count]):
        for key in ('objective','gradient_norm_before_clip','routing_seed'):
            if left[key]!=right[key]:
                raise AssertionError(f'training prefix diverged from reproducibility oracle: {name} {key}')


def trial(out, variant, lr, seed=42, steps=64, fingerprint=False):
    out.mkdir(parents=True,exist_ok=True)
    tag = f'{variant}-lr{lr:g}-seed{seed}-steps{steps}'
    path = out / (tag + '.json')
    if path.exists():
        report = json.loads(path.read_text())
        if report.get('status') == 'passed' and 'external_training_power' in report:
            validate_learning_protocol(report,fingerprint)
            return report
        raise RuntimeError(f'failed/incomplete trial already exists: {path}')
    command = [sys.executable, '-m', 'replication.compare_muon_deterministic',
               '--weights', '/mnt/fast/models/mini-AGI/undertrained/weights',
               '--text', '/mnt/fast/models/mini-AGI/evaluation-data/TinyStoriesV2-GPT4-valid.txt',
               '--optimizer', variant, '--lr', str(lr), '--seed', str(seed),
               '--steps', str(steps), '--out', str(path)]
    if fingerprint:
        command.append('--fingerprint')
    print(json.dumps(dict(starting=tag)), flush=True)
    with PowerMonitor() as monitor:
        with path.with_suffix('.log').open('w') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    monitor.save(path.with_suffix('.external-power.json'), command=command, returncode=code)
    if code:
        raise RuntimeError(f'trial failed: {tag}; see log')
    report = json.loads(path.read_text())
    validate_learning_protocol(report,fingerprint)
    report['external_training_power'] = summarize_samples(monitor.samples, report['training_start'],
                                                         report['training_end'], steps*512)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(finished=tag, tune_ce=report['final_tune']['mean_ce'],
                          test_ce=report['final_test']['mean_ce'],
                          seconds=report['training_end']-report['training_start'],
                          watts=report['external_training_power']['average_board_power_w'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/muon-screen'))
    parser.add_argument('--phase', choices=('repro','smoke', 'grid', 'low-grid', 'confirm', 'long-grid', 'long-confirm', 'latency', 'latency-large'), required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['PATH'] = str(ROOT/'.venv/bin') + os.pathsep + os.environ['PATH']
    os.environ['MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH'] = '1'
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '2'
    variants = ('adamw', 'gefen', *VARIANTS)
    if args.phase == 'repro':
        first_gradients=[]
        for name in variants:
            a=trial(args.out/'a',name,3e-5,steps=8,fingerprint=True)
            b=trial(args.out/'b',name,3e-5,steps=8,fingerprint=True)
            for key in ('initial_tune','initial_test','final_tune','final_test','first_gradient_sha256','final_model_sha256','final_optimizer_sha256','final_host_experts_sha256','final_slots'):
                if a[key]!=b[key]:
                    raise AssertionError(f'cross-process reproducibility failed: {name} {key}')
            for left,right in zip(a['trace'],b['trace']):
                for key in ('objective','gradient_norm_before_clip','routing_seed'):
                    if left[key]!=right[key]:
                        raise AssertionError(f'cross-process training trace differed: {name} {key}')
            first_gradients.append(a['first_gradient_sha256'])
            print(json.dumps({'reproducible':name}),flush=True)
        if len(set(first_gradients))!=1:
            raise AssertionError('candidates did not receive bit-identical initial gradients')
        (args.out/'repro-passed.json').write_text(json.dumps({'variants':variants,'initial_gradients_identical_across_candidates':True,
                                                            'two_independent_processes_each':True},indent=2)+'\n')
    elif args.phase == 'smoke':
        for name in variants:
            trial(args.out, name, .001 if name == 'muon_plus' else .0001, steps=8)
    elif args.phase in ('grid', 'low-grid'):
        rates = (3e-6, 1e-5) if args.phase == 'low-grid' else (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
        for lr in rates:
            # Interleave candidates across rates, rather than timing an entire
            # candidate in one thermal/background-activity block.
            for name in variants:
                trial(args.out, name, lr)
        if args.phase == 'low-grid':
            trial(args.out, 'adamw', 0.)  # auxiliary-only adaptation control
    elif args.phase == 'long-grid':
        for rate in (3e-6, 3e-5, 1e-4):
            for name in variants:
                trial(args.out,name,rate,steps=256)
        short_path=args.out/'selected-rates.json'
        short = json.loads(short_path.read_text()) if short_path.exists() else {}
        for name,rate in short.items():
            if rate not in (3e-6,3e-5,1e-4):
                trial(args.out,name,rate,steps=256)
        trial(args.out,'adamw',0.,steps=256)
    elif args.phase in ('confirm','long-confirm'):
        horizon = 256 if args.phase == 'long-confirm' else 64
        selected = {}
        for name in variants:
            candidates = [json.loads(p.read_text()) for p in args.out.glob(f'{name}-lr*-seed42-steps{horizon}.json')]
            if len(candidates) < (3 if horizon==256 else 6) or any(r['status'] != 'passed' for r in candidates):
                raise RuntimeError(f'incomplete grid for {name}')
            for attempt in range(7):
                best = min(candidates, key=lambda r:r['final_tune']['mean_ce'])
                rate = best['configuration']['lr']
                tested = sorted(r['configuration']['lr'] for r in candidates)
                if rate == 0 or rate not in (tested[0], tested[-1]):
                    break
                if rate == tested[0] and rate > 1e-8:
                    extra_rate = float(f'{rate/3:.1g}')
                elif rate == tested[-1] and rate < .1:
                    extra_rate = float(f'{rate*3:.1g}')
                else:
                    raise RuntimeError(f'unbracketed best LR for {name}; further tuning required')
                print(json.dumps({'extending_edge':name,'lr':extra_rate}),flush=True)
                candidates.append(trial(args.out,name,extra_rate,steps=horizon))
            else:
                raise RuntimeError(f'could not bracket learning rate for {name}')
            selected[name] = best['configuration']['lr']
        selection_file = 'selected-rates-long.json' if horizon==256 else 'selected-rates.json'
        (args.out/selection_file).write_text(json.dumps(selected, indent=2)+'\n')
        # No test loss is consulted for selection. All candidates get both
        # confirmation data orders, including those that lost the short grid.
        for seed in (43, 44):
            for name in variants:
                trial(args.out, name, selected[name], seed=seed, steps=256)
    else:
        for name in variants:
            prefix = 'latency-large' if args.phase == 'latency-large' else 'latency'
            path = args.out / f'{prefix}-{name}.json'
            if path.exists():
                raise RuntimeError(f'latency result already exists: {path}')
            command = [sys.executable, '-m', 'replication.probe_muon_variants',
                       '--optimizer', name, '--out', str(path),'--deterministic']
            if args.phase == 'latency-large':
                command += ['--experts','4','--hidden','4096','--width','16384']
            print(json.dumps({'starting_latency':name}), flush=True)
            with PowerMonitor() as monitor:
                with path.with_suffix('.log').open('w') as log:
                    code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            monitor.save(path.with_suffix('.external-power.json'), command=command, returncode=code)
            if code:
                if args.phase == 'latency-large' and 'OutOfMemoryError' in path.with_suffix('.log').read_text():
                    path.write_text(json.dumps({'status':'out_of_memory', 'optimizer':name,
                                                'allocator_fraction':.75,
                                                'geometry':{'experts':4,'hidden':4096,'width':16384},
                                                'external_telemetry':str(path.with_suffix('.external-power.json'))},indent=2)+'\n')
                    print(json.dumps({'large_geometry_oom':name}),flush=True)
                    continue
                raise RuntimeError(f'latency probe failed: {name}')
            report = json.loads(path.read_text())
            report['external_power'] = summarize_samples(monitor.samples, report['start_monotonic'], report['end_monotonic'])
            report['external_board_joules_per_step'] = report['external_power']['estimated_board_energy_j']/report['iterations']
            path.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'finished_latency':name,'median_ms':report['median_step_seconds']*1000}),flush=True)


if __name__ == '__main__':
    main()

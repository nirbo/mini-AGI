"""Serial GPU screen with external telemetry; never writes model checkpoints."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from replication.gpu_monitor import PowerMonitor, summarize_samples
from replication.muon_candidates import VARIANTS, ROOT


def trial(out, variant, lr, seed=42, steps=64):
    tag = f'{variant}-lr{lr:g}-seed{seed}-steps{steps}'
    path = out / (tag + '.json')
    if path.exists():
        report = json.loads(path.read_text())
        if report.get('status') == 'passed' and 'external_training_power' in report:
            return report
        raise RuntimeError(f'failed/incomplete trial already exists: {path}')
    command = [sys.executable, '-m', 'replication.compare_muon_training',
               '--weights', '/mnt/fast/models/mini-AGI/undertrained/weights',
               '--text', '/mnt/fast/models/mini-AGI/evaluation-data/TinyStoriesV2-GPT4-valid.txt',
               '--optimizer', variant, '--lr', str(lr), '--seed', str(seed),
               '--steps', str(steps), '--out', str(path)]
    print(json.dumps(dict(starting=tag)), flush=True)
    with PowerMonitor() as monitor:
        with path.with_suffix('.log').open('w') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    monitor.save(path.with_suffix('.external-power.json'), command=command, returncode=code)
    if code:
        raise RuntimeError(f'trial failed: {tag}; see log')
    report = json.loads(path.read_text())
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
    parser.add_argument('--phase', choices=('smoke', 'grid', 'low-grid', 'confirm', 'latency', 'latency-large'), required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['PATH'] = str(ROOT/'.venv/bin') + os.pathsep + os.environ['PATH']
    os.environ['MINAGI_GEFEN_ALLOW_CUDA_MINOR_MISMATCH'] = '1'
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '2'
    variants = ('adamw', 'gefen', *VARIANTS)
    if args.phase == 'smoke':
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
    elif args.phase == 'confirm':
        selected = {}
        for name in variants:
            candidates = [json.loads(p.read_text()) for p in args.out.glob(f'{name}-lr*-seed42-steps64.json')]
            if len(candidates) < 6 or any(r['status'] != 'passed' for r in candidates):
                raise RuntimeError(f'incomplete grid for {name}')
            best = min(candidates, key=lambda r:r['final_tune']['mean_ce'])
            tested = sorted(r['configuration']['lr'] for r in candidates)
            if best['configuration']['lr'] in (tested[0], tested[-1]):
                raise RuntimeError(f'unbracketed best LR for {name}; extend the grid before confirmation')
            selected[name] = best['configuration']['lr']
        (args.out/'selected-rates.json').write_text(json.dumps(selected, indent=2)+'\n')
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
                       '--optimizer', name, '--out', str(path)]
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

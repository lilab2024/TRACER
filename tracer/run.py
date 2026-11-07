"""Run simulation, preprocessing, five independent CH-GAIL repairs and metrics."""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stages.step01_simulate import simulate
from stages.step02_trim import trim
from stages.step03_beta import compute, normal_bounds
from stages.step04_label import label, split_training
from stages import step05_repair_core as core
from stages.step06_metrics import evaluate, summarize

ROOT = Path(__file__).resolve().parent
SCENARIOS = ['ba', 'dpa', 'ma', 'pa', 'fa']
STATE_COLS = ['Spacing', 'Speed_Follower', 'Relative_speed']


def save(frame, path):
    frame.to_csv(path, sep='\t', index=False, float_format='%.6f')


def read_trajectory(path, columns):
    frame = pd.read_csv(path, sep=r'\s+')
    if not set(columns).issubset(frame.columns):
        raise ValueError(f'Missing columns in {path}: {columns}')
    frame = frame[columns].copy()
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f'Nonfinite data in {path}')
    if frame.Time.tolist() != list(range(len(frame))):
        raise ValueError(f'Expected ordered, consecutive integer Time starting at zero: {path}')
    frame['Time'] = frame.Time.astype(int)
    return frame


def run(args):
    config = json.loads((ROOT/'config.json').read_text())
    torch.set_num_threads(config['torch_threads'])
    out = Path(args.output).resolve() if args.output else ROOT/'outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    config.update({'smoke_test': args.smoke, 'prepare_only': args.prepare_only,
                   'scenarios': SCENARIOS if args.scenario == 'all' else [args.scenario]})
    if args.normal_target:
        config['input_files'] = {name: {'path': str(Path(path).resolve()),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for name, path in [('normal_target', args.normal_target), ('normal_leader', args.normal_leader), ('attack_target', args.attack_target)]}
    (out/'config_used.json').write_text(json.dumps(config, indent=2))
    normal_dir = out/'normal'; normal_dir.mkdir()
    if args.normal_target:
        raw = read_trajectory(args.normal_target, ['Time'] + STATE_COLS)
        lead_raw = read_trajectory(args.normal_leader, ['Time', 'Speed_Leader'])
        if raw.Time.tolist() != lead_raw.Time.tolist():
            raise ValueError('Normal target and leader Time must match')
    else:
        raw, lead_raw = simulate('normal')
    save(raw, normal_dir/'01_target_raw.txt'); save(lead_raw, normal_dir/'01_leader_raw.txt')
    normal, leader = trim(raw), trim(lead_raw)
    save(normal, normal_dir/'02_target_trimmed.txt'); save(leader, normal_dir/'02_leader_trimmed.txt')
    normal_beta = compute(normal); save(normal_beta, normal_dir/'03_target_beta.txt')
    bounds = normal_bounds(normal_beta)
    pd.DataFrame([{'beta': c, 'minimum': b[0], 'maximum': b[1]} for c,b in bounds.items()]).to_csv(normal_dir/'04_beta_bounds.txt', sep='\t', index=False, float_format='%.6f')
    all_summary = []
    for scenario in config['scenarios']:
        folder = out/scenario; folder.mkdir()
        if args.attack_target:
            raw = read_trajectory(args.attack_target, ['Time'] + STATE_COLS)
            if not set(raw.Time).issubset(set(lead_raw.Time)):
                raise ValueError('Normal reference must cover all attack Time values')
        else:
            raw, _ = simulate(scenario)
        save(raw, folder/'01_target_raw.txt')
        cropped = trim(raw); save(cropped, folder/'02_target_trimmed.txt')
        beta = compute(cropped); save(beta, folder/'03_target_beta.txt')
        labeled = label(beta, bounds); save(labeled, folder/'04_target_labeled.txt')
        expert, abnormal = split_training(labeled)
        save(expert, folder/'05_training_normal_before_attack.txt')
        save(abnormal, folder/'05_abnormal_input.txt')
        metadata = {'training_rows':len(expert), 'abnormal_rows':len(abnormal),
                    'evaluation_rows':int(abnormal.Time.between(3600,3900).sum())}
        (folder/'sample_counts.json').write_text(json.dumps(metadata, indent=2))
        print(scenario, metadata, flush=True)
        if args.prepare_only: continue
        seed = config['training_seed']
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        core.GAIL_EPOCHS = 1 if args.smoke else config['gail_epochs']
        metrics=[]
        for iteration in range(1,6):
            if abnormal.empty:
                repaired = pd.DataFrame(columns=['Time','Repaired_Spacing','Repaired_Speed','Repaired_Relative_speed'])
            else:
                model = core.HGAIL()
                model.train_gailpair(expert[STATE_COLS].values, expert[['Speed_Follower']].values)
                states, actions = core.repair_trajectory(model, abnormal[STATE_COLS].values, leader.Speed_Leader.values)
                repaired = pd.DataFrame(states, columns=['Repaired_Spacing','Repaired_Speed','Repaired_Relative_speed'])
                repaired.insert(0,'Time', abnormal.Time.to_numpy())
                for c in STATE_COLS: repaired['Original_'+c] = abnormal[c].to_numpy()
                torch.save(model.policy.state_dict(), folder/f'06_policy_run_{iteration:02d}.pt')
            save(repaired, folder/f'06_repaired_run_{iteration:02d}.txt')
            try:
                values = evaluate(repaired, normal)
                values['status'] = 'valid' if values['evaluation_rows'] else 'no_evaluation_rows'
            except ValueError as error:
                values = {'evaluation_rows': metadata['evaluation_rows'], 'status': 'invalid', 'reason': str(error)}
            metrics.append({'run':iteration, **values})
            print(f'{scenario}: repair {iteration}/5 complete' if not abnormal.empty else f'{scenario}: run {iteration}/5 skipped (no abnormal rows)', flush=True)
        save(pd.DataFrame(metrics), folder/'07_metrics_each_run.txt')
        summary=summarize(metrics);save(summary, folder/'08_metrics_mean_std.txt')
        if not summary.empty:
            summary.insert(0,'scenario',scenario); all_summary.append(summary)
    if all_summary: save(pd.concat(all_summary,ignore_index=True),out/'09_all_metrics_mean_std.txt')
    print(f'Output: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['all']+SCENARIOS, default='all')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One training epoch, for wiring checks only')
    parser.add_argument('--output', help='New output directory; must not already exist')
    parser.add_argument('--normal-target', type=Path, help='Untrimmed normal follower TXT')
    parser.add_argument('--normal-leader', type=Path, help='Untrimmed normal leader TXT')
    parser.add_argument('--attack-target', type=Path, help='Untrimmed attack follower TXT')
    args = parser.parse_args()
    supplied = [args.normal_target, args.normal_leader, args.attack_target]
    if any(supplied) and (not all(supplied) or args.scenario == 'all'):
        parser.error('Provide all three input files and a single --scenario')
    run(args)

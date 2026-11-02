"""Run the original DPA/MA pipeline with automatic file routing."""
import argparse
import json
from pathlib import Path
from workflow import SCENARIOS, run

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--scenario', type=str.lower, choices=['all', *SCENARIOS],
                        help='Run both attacks (all), DPA, or MA; default: config setting')
    parser.add_argument('--quick', action='store_true', help='One run, one epoch per stage, for smoke testing')
    parser.add_argument('--prepare-only', action='store_true', help='Export trajectories, beta, bounds and flags only')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    scenario = args.scenario or cfg.get('scenario', 'all').lower()
    if scenario not in ['all', *SCENARIOS]:
        raise ValueError('scenario must be all, dpa, or ma')
    if args.quick:
        cfg.update(runs=1, selection_keep=1, gen_epochs=1, bc_epochs=1, gail_epochs=1, torch_threads=1)
    for key in ['runs', 'gen_epochs', 'bc_epochs', 'gail_epochs']:
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    keep = cfg.get('selection_keep', cfg['runs'])
    if not isinstance(keep, int) or isinstance(keep, bool) or not 1 <= keep <= cfg['runs']:
        raise ValueError('selection_keep must be between 1 and runs')
    if not isinstance(cfg['trim_start'], int) or not 0 <= cfg['trim_start'] < 2800:
        raise ValueError('trim_start must be in [0, 2799] because the original bounds stage slices again')
    base = Path(cfg['output_dir'])
    if not base.is_absolute():
        base = ROOT / base
    mode = 'prepare' if args.prepare_only else ('quick' if args.quick else 'full')
    scenarios = list(SCENARIOS) if scenario == 'all' else [scenario]
    for attack in scenarios:
        out = base / attack / f"best_{keep}_of_{cfg['runs']}" / mode
        run({**cfg, 'scenario': attack}, out, args.prepare_only)


if __name__ == '__main__':
    main()

# TRACER

An end-to-end Python workflow for detecting and repairing anomalous adaptive cruise control (ACC) trajectories using CRIL / CH-GAIL. It includes a normal driving scenario and five attack scenarios: BA, DPA, MA, PA, and FA.

## Workflow

1. Simulate trajectories for the ACC vehicle (array index 1) and its leader (array index 0). Save both vehicles for the normal scenario and the ACC vehicle for each attack.
2. Discard the first 2,000 time steps, keeping the original `Time` values.
3. Compute beta1, beta2, and beta3. Estimate normal intervals from the minimum and maximum values across the complete trimmed normal trajectory.
4. Label a row abnormal if **beta1 or beta3** falls outside its normal interval. Beta2 is saved but does not determine the label. Interval boundaries are included as normal.
5. Train only on normal-labeled rows before the attack starts (`Time < 3600`). Perform five independent training and repair runs and save each result.
6. Evaluate repaired abnormal rows against the normal ACC trajectory, matched by `Time`. Report spacing and speed MAE, RMSE, and MAPE, followed by the mean and population standard deviation across all five runs.

## Installation

Use Python 3.11:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run all five attack scenarios:

```bash
python run.py
```

Run one scenario:

```bash
python run.py --scenario fa
```

Generate trajectories, beta values, labels, and training inputs without training:

```bash
python run.py --scenario fa --prepare-only
```

Use three existing, untrimmed trajectory files instead of simulation:

```bash
python run.py --scenario fa \
  --normal-target /path/to/normal_target.txt \
  --normal-leader /path/to/normal_leader.txt \
  --attack-target /path/to/fa_target.txt
```

Target files require `Time`, `Speed_Follower`, `Relative_speed`, and `Spacing`. The leader file requires `Time` and `Speed_Leader`. Files must be whitespace-delimited, with consecutive integer time steps starting at zero. Normal target and leader times must match and cover the attack trajectory. Files may have different lengths; each is trimmed once at step 2000. Input paths and SHA256 hashes are recorded.

## Evaluation

**Only repaired abnormal rows within the evaluation window are scored.** Normal-labeled rows are excluded; they are not added to produce a fixed-length evaluation set.

The original metric script uses `3600 <= Time <= 3900`. These inclusive endpoints contain 301 possible time indices; the actual evaluation count depends on the abnormal labels. The simulated attack itself is active for `3600 <= Time < 3900` (300 steps). The original evaluation endpoints are preserved here.

MAPE is reported as a percentage using the normal reference value plus `1e-6` as the denominator. Standard deviation uses `ddof=0`. All five runs are included without ranking or selection. If any run is invalid or has no evaluation rows, the five-run aggregate is marked incomplete instead of averaging only the successful runs. If no anomalies are detected, training and repair are skipped and repair files are empty.

## Configuration and method

`config.json` defaults to 60 epochs per GAIL pair, five independent repair runs, and no explicit training seed (`training_seed: null`). The simulation retains its original NumPy seed of 42. Training results therefore vary between executions. An integer training seed, if supplied, is set once before the five runs for each scenario.

ACC and beta parameters are `v0=39.12`, `tau=0.96`, `s0=10`, `delta=20`, `a=1.18`, and `b=8.9`. The simulation uses six vehicles, 4,800 steps, and a 0.1-second integration step. Simulated trajectory inputs retain three decimal places; beta values retain six.

Preprocessing retains the specified original beta3 term `tau - Relative_speed / (2 * sqrt(a * b))`. The active CH-GAIL classes, loss functions, state noise, and repair recurrence are preserved from the original five-run repair script. Two GAIL pairs are trained per run; the final policy is taken from the pair with state noise.

The inherited repair recurrence consumes the trimmed normal leader speed by abnormal-row position rather than by `Time`, and treats all abnormal rows as a continuous sequence, including gaps. Its output relative speed is leader minus follower, while the simulation input uses follower minus leader. These conventions are retained to preserve the original repair method. Metric alignment uses the original `Time` values.

## Files and outputs

```text
tracer/
  run.py                  # Unified entry point
  run.command             # Optional macOS launcher
  config.json             # Training configuration
  requirements.txt
  simulators/             # Normal and attack simulations
  stages/
    step01_simulate.py
    step02_trim.py
    step03_beta.py
    step04_label.py
    step05_repair_core.py
    step06_metrics.py
```

Each execution creates a new `outputs/<timestamp>/` directory containing:

- Normal raw/trimmed trajectories, beta values, and interval bounds.
- Per-attack raw/trimmed trajectories, beta values, labels, training rows, and abnormal inputs.
- Five repaired trajectory files and, when training occurs, five policy weight files.
- Per-run metrics, five-run summaries, sample counts, and the configuration used.

`outputs/` is generated automatically and excluded by `.gitignore`. It can be removed after saving any results you need. No generated experiment results are required to run the repository.

## Verification

The six simulation outputs were checked against the pre-cleanup simulation code. Active CH-GAIL classes and the repair recurrence were verified to have identical Python syntax trees to the original implementation. Label logic, pre-attack training selection, and metric matching by `Time` were checked. For a short integration check, use `--smoke`; it runs one training epoch and must not be used as a research result.

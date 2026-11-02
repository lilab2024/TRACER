# TRACER: DPA and MA Experiments

An end-to-end runner for normal trajectory simulation, beta-based anomaly detection, CRIL trajectory repair, and error evaluation. Only DPA and MA attacks are included.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

```bash
python run.py                 # DPA and MA, sequentially
python run.py --scenario dpa  # DPA only
python run.py --scenario ma   # MA only
```

On macOS, double-click `run.command` after installing dependencies. It runs both scenarios by default. Change `scenario` in `config.json` to `dpa` or `ma` to select one.

```bash
python run.py --quick         # One run, one epoch per stage; a smoke test only
python run.py --prepare-only  # Trajectories, beta, bounds, and anomaly flags
```

## Experiment settings

- ACC vehicle index: **1**. Leader index: **0**.
- Simulation: 4,800 steps, 0.1 seconds per step, six vehicles on a 300 m ring.
- Normal, DPA, MA, and beta parameters: `[v0, T, s0, delta, a, b] = [39.12, 0.96, 10.00, 20.00, 1.18, 8.90]`.
- Attack condition: `3600 <= t < 3900`. Evaluation window: `3600 <= Time <= 3900`.
- Each scenario generates one normal simulation and one attack simulation, followed by **10 independent training/repair runs on those trajectories**.
- Generator: 60 epochs; BC: 80 epochs; each GAIL stage: 60 epochs.
- The simulation's original `np.random.seed(42)` is retained. **The runner does not set an additional CRIL training seed.** Each training run creates a new model.

## Selecting five runs

Runs are ranked by `(speed_mape + spacing_mape) / 2`, ascending. The five lowest-scoring complete runs are selected. Ties are resolved by ascending run ID. All six reported metrics use the **same selected runs**. Standard deviation uses `ddof=0`.

The selected-five statistics are selected-subset results on the evaluation window, not an unbiased estimate of all ten runs or an independent test score. Every repair file and per-run metric is retained, together with statistics for **all ten runs**, the complete ranking, and the selected run IDs.

Edit `runs`, `selection_keep`, or training epochs in `config.json` as needed. Each execution starts a new experiment and overwrites files at the corresponding output path.

## Outputs

Default directories:

```text
outputs/dpa/best_5_of_10/full/
outputs/ma/best_5_of_10/full/
```

| File | Contents |
| --- | --- |
| `normal_acc_1_raw.txt`, `normal_leader_0_raw.txt` | Normal simulation exports |
| `dpa_acc_1_raw.txt` or `ma_acc_1_raw.txt` | Attack simulation export |
| `normal_acc_1.txt`, `normal_leader_0.txt`, `<scenario>_acc_1.txt` | Trajectories with `Time >= 2000` |
| `normal_beta.txt`, `<scenario>_beta.txt`, `beta_bounds.txt` | Three beta values and normal bounds |
| `<scenario>_beta_flag.txt` | Anomaly flags |
| `repaired_run_1.txt` to `repaired_run_10.txt` | All repair outputs |
| `metrics_per_run.csv` | All ten runs and all six metrics |
| `metrics_all_runs_summary.csv`, `.txt` | All-ten mean and standard deviation |
| `selection_ranking.csv`, `selection_used.json` | Scores, ranks, selection flags, and selected IDs |
| `metrics_selected_runs.csv` | Per-run metrics for the selected five |
| `metrics_summary.csv`, `.txt` | Selected-five mean and standard deviation |
| `config_used.json`, `source_sha256.json` | Configuration and source checksums |

Quick runs use `best_1_of_1/quick/`. Preparation-only runs use `best_5_of_10/prepare/` under the default configuration. Generated outputs, virtual environments, and caches are ignored by Git.

## Source layout and retained behavior

```text
run.py          CLI entry point
run.command     macOS launcher
workflow.py     File routing, configuration, and stage execution
selection.py    Run ranking and summary statistics
stages/         Normal, DPA, MA, beta, bounds, flags, CRIL, and metrics
```

The stage files are required at runtime. They contain the original numerical logic with English filenames, notebook comments removed, and unrelated post-export simulation plotting omitted. Paths are connected automatically. No simulation or repair formulas were changed for this edition.

For compatibility with the existing workflow:

- Trajectories retain the original three-decimal export precision; beta exports retain six decimals and repaired exports four decimals.
- After the `Time >= 2000` cut, the original bounds stage still applies `iloc[2000:]`: bounds therefore use steps 4000–4799.
- `flag1` uses beta1 and `flag2` uses beta3. Original status values and stage log messages are retained.
- CRIL still processes the filtered abnormal rows with its original recurrence, leader-row indexing, relative-speed convention, BC/GAIL flow, and repaired-beta calculation.
- Metrics use abnormal timestamps present in the repair output within the evaluation window. Normal timestamps are not added to the repair file.
- DPA retains its original random-number draw sequence, which differs from the normal simulation. Equal simulation seeds do not imply identical pre-attack perturbations for DPA.
- Original metric-stage console logs are stored separately in `metrics_all_runs_original.txt` and `metrics_selected_original.txt`. Their historical fixed run-count wording comes from the stage source; use the configuration and summary files for actual counts.

No historical outputs, notebooks, archives, diagnostic experiments, or BA attack code are included.

"""Execute archived scripts, overriding only I/O and explicitly configured parameters."""
import ast
import contextlib
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from selection import rank_runs, summarize

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "stages"
SCENARIOS = {"dpa": "dpa.py", "ma": "ma.py"}


def source(pattern):
    return next(SOURCE.glob(pattern))


def execute(pattern, overrides=None, replacements=None, run_name="__main__"):
    path = source(pattern)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    overrides = overrides or {}
    replacements = replacements or {}
    class Wiring(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            key = ast.unparse(node.targets[0])
            if key in overrides:
                node.value = ast.parse(repr(overrides[key]), mode="eval").body
            return node
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in replacements:
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            return node
        def visit_Call(self, node):
            self.generic_visit(node)
            # pandas >= 2.2 removed delim_whitespace; same whitespace parser.
            for kw in node.keywords:
                if kw.arg == "delim_whitespace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    kw.arg = "sep"
                    kw.value = ast.Constant(r"\s+")
            return node
    tree = ast.fix_missing_locations(Wiring().visit(tree))
    namespace = {"__name__": run_name, "__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def read(path):
    return pd.read_csv(path, sep=r"\s+")


def trim(input_path, output_path, start):
    df = read(input_path)
    df = df[df.Time >= start].reset_index(drop=True)
    df.to_csv(output_path, sep="\t", index=False, float_format="%.3f")


def run(cfg, out, prepare_only=False):
    scenario = cfg.get("scenario", "dpa").lower()
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown attack scenario: {scenario}")
    cfg = {**cfg, "scenario": scenario}
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: str((out / (name + ".txt")).resolve()) for name in [
        "normal_acc_1_raw", "normal_leader_0_raw", f"{scenario}_acc_1_raw", f"{scenario}_leader_unused",
        "normal_acc_1", "normal_leader_0", f"{scenario}_acc_1", "normal_beta", f"{scenario}_beta", f"{scenario}_beta_flag"]}
    (out / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob("*.py")}
    (out / "source_sha256.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2))
    print(f"[1/4] Simulating normal/{scenario.upper()} once each", flush=True)
    for pattern, kind in [("normal.py", "normal"), (SCENARIOS[scenario], scenario)]:
        execute(pattern, {"output_file": paths[f"{kind}_acc_1_raw"],
                          "output_file_leader": paths["normal_leader_0_raw" if kind == "normal" else f"{scenario}_leader_unused"]})
    Path(paths[f"{scenario}_leader_unused"]).unlink()  # Attack leader is unused; preserve original exporter execution.
    for name in ["normal_acc_1", "normal_leader_0", f"{scenario}_acc_1"]:
        trim(paths[name + "_raw"], paths[name], cfg["trim_start"])
    print("[2/4] Computing beta, bounds and anomaly flags from TXT files", flush=True)
    for kind in ["normal", scenario]:
        execute("beta.py", {"trajectory_file": paths[f"{kind}_acc_1"], "output_file": paths[f"{kind}_beta"]})
    ns = execute("beta_bounds.py", {"file_path": paths["normal_beta"]})
    bounds = {f"beta{i}": [float(ns[f"min_beta{i}"]), float(ns[f"max_beta{i}"])] for i in range(1, 4)}
    if not np.isfinite(list(bounds.values())).all():
        raise ValueError("No finite bounds remain after the original iloc[2000:] slice")
    (out / "beta_bounds.txt").write_text("\n".join(f"{key} {lo:.9f} {hi:.9f}" for key, (lo, hi) in bounds.items()) + "\n")
    flag_overrides = {"file_path": paths[f"{scenario}_beta"], "output_path": paths[f"{scenario}_beta_flag"]}
    cril_overrides = {}
    if cfg["bounds_source"] == "normal":
        selected = cfg.get("custom_bounds") or bounds
        flag_overrides.update({"(beta1_min, beta1_max)": selected["beta1"], "(beta3_min, beta3_max)": selected["beta3"]})
        cril_overrides.update({"(BETA1_MIN, BETA1_MAX)": selected["beta1"], "(BETA3_MIN, BETA3_MAX)": selected["beta3"]})
    elif cfg["bounds_source"] != "saved_script":
        raise ValueError("bounds_source must be normal or saved_script")
    with (out / "flag_stage.log").open("w") as log, contextlib.redirect_stdout(log):
        execute("flag_anomalies.py", flag_overrides)
    flagged = read(paths[f"{scenario}_beta_flag"])
    bad = (flagged.flag1 != "正常") | (flagged.flag2 != "正常")
    if bad.all() or not bad.any():
        raise ValueError("CRIL requires both normal expert samples and abnormal samples")
    print(f"Abnormal rows: {int(bad.sum())}; normal expert rows: {int((~bad).sum())}", flush=True)
    (out / "thresholds_used.json").write_text(json.dumps({
        "bounds_source": cfg["bounds_source"], "computed_normal_bounds": bounds,
        "flag_bounds": {"beta1": flag_overrides.get("(beta1_min, beta1_max)", [0.002590, 0.078830]),
                        "beta3": flag_overrides.get("(beta3_min, beta3_max)", [-0.071054, -0.011188])},
        "cril_output_flag_bounds": {"beta1": cril_overrides.get("(BETA1_MIN, BETA1_MAX)", [0.061533, 0.078499]),
                                    "beta3": cril_overrides.get("(BETA3_MIN, BETA3_MAX)", [-0.075789, -0.058077])}}, indent=2))
    if prepare_only:
        return
    window_bad = bad & flagged.Time.between(3600, 3900)
    if not window_bad.any():
        raise ValueError("No abnormal rows exist in the 3600-3900 evaluation window; no metrics can be computed")
    repaired_files = []
    for i in range(cfg["runs"]):
        print(f"[3/4] CRIL training and repair {i+1}/{cfg['runs']}", flush=True)
        import torch
        if cfg.get("torch_threads") is not None:
            torch.set_num_threads(cfg["torch_threads"])
        repaired = str((out / f"repaired_run_{i+1}.txt").resolve())
        execute("cril.py", {**cril_overrides, "OUTPUT_PATH": repaired, "LEADER_FILE": paths["normal_leader_0"],
                         "GEN_EPOCHS": cfg["gen_epochs"], "BC_EPOCHS": cfg["bc_epochs"], "GAIL_EPOCHS": cfg["gail_epochs"]},
                {"attack_beta_flag.txt": paths[f"{scenario}_beta_flag"]})
        repaired_files.append(repaired)
    print("[4/4] Computing metrics for abnormal timestamps present in repaired files", flush=True)
    with (out / "metrics_all_runs_original.txt").open("w") as log, contextlib.redirect_stdout(log):
        ns = execute("metrics.py", {"normal_file": paths["normal_beta"], "repaired_files": repaired_files})
    columns = ["speed_mae", "speed_rmse", "speed_mape", "spacing_mae", "spacing_rmse", "spacing_mape"]
    per_run = pd.DataFrame({key: ns[key + "_list"] for key in columns})
    if len(per_run) != cfg["runs"] or not np.isfinite(per_run.to_numpy()).all():
        raise ValueError("Not all runs produced finite metrics in the evaluation window")
    per_run.insert(0, "run", range(1, len(per_run)+1))
    per_run.to_csv(out / "metrics_per_run.csv", index=False)
    all_summary = summarize(per_run)
    all_summary.to_csv(out / 'metrics_all_runs_summary.csv', index_label='metric')
    (out / 'metrics_all_runs_summary.txt').write_text(
        f'All {len(per_run)} training/repair runs: mean and population std (ddof=0)\n' + all_summary.to_string() + '\n')
    keep = cfg.get('selection_keep', len(per_run))
    metric = cfg.get('selection_metric', 'mean_mape')
    ranked = rank_runs(per_run, keep, metric)
    ranked.to_csv(out / 'selection_ranking.csv', index=False)
    selected = ranked[ranked.selected].copy()
    selected.to_csv(out / 'metrics_selected_runs.csv', index=False)
    selected_files = [repaired_files[int(i)-1] for i in selected['run']]
    # Execute the original metric script on the SAME selected runs for every metric.
    with (out / 'metrics_selected_original.txt').open('w') as log, contextlib.redirect_stdout(log):
        selected_ns = execute('metrics.py', {'normal_file': paths['normal_beta'], 'repaired_files': selected_files})
    for key in columns:
        np.testing.assert_allclose(selected_ns[key+'_list'], selected[key].to_numpy(), rtol=1e-10, atol=1e-12)
    summary = summarize(selected)
    summary.to_csv(out / 'metrics_summary.csv', index_label='metric')
    (out / 'selection_used.json').write_text(json.dumps({
        'total_runs': len(per_run), 'keep_runs': keep, 'metric': metric,
        'formula': '(speed_mape + spacing_mape) / 2' if metric == 'mean_mape' else metric,
        'tie_break': 'run ascending', 'selected_run_ids': selected['run'].astype(int).tolist(),
        'evaluation_window': [3600,3900], 'std_ddof': 0,
        'result_type': 'selected_subset' if keep < len(per_run) else 'all_runs'
    }, indent=2))
    label = f'Selected {keep} of {len(per_run)} runs by ascending {metric}; all metrics use the same subset, std ddof=0'
    (out / 'metrics_summary.txt').write_text(label + '\n' + summary.to_string() + '\n')
    print(label)
    print('Selected run IDs: ', selected['run'].astype(int).tolist())
    print(summary.to_string())
    print(f'Results: {out}')

"""One shared ranking for selecting complete runs, not individual metrics."""
import numpy as np

METRICS = ['speed_mae', 'speed_rmse', 'speed_mape', 'spacing_mae', 'spacing_rmse', 'spacing_mape']


def rank_runs(per_run, keep, metric='mean_mape'):
    if not isinstance(keep, int) or isinstance(keep, bool) or not 1 <= keep <= len(per_run):
        raise ValueError('selection_keep must be between 1 and the actual number of runs')
    if per_run['run'].duplicated().any():
        raise ValueError('Duplicate run IDs are not allowed')
    if not np.isfinite(per_run[METRICS].to_numpy()).all():
        raise ValueError('Metrics contain NaN/Inf; failed runs cannot be silently removed')
    ranked = per_run.copy()
    if metric == 'mean_mape':
        ranked['selection_score'] = (ranked.speed_mape + ranked.spacing_mape) / 2
    elif metric in METRICS:
        ranked['selection_score'] = ranked[metric]
    else:
        raise ValueError(f'Unknown selection metric: {metric}')
    ranked = ranked.sort_values(['selection_score', 'run'], kind='stable').reset_index(drop=True)
    ranked['rank'] = np.arange(1, len(ranked) + 1)
    ranked['selected'] = ranked['rank'] <= keep
    return ranked


def summarize(per_run):
    values = per_run[METRICS]
    import pandas as pd
    return pd.DataFrame({'mean': values.mean(), 'std': values.std(ddof=0)})

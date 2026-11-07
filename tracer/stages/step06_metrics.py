import numpy as np
import pandas as pd


def evaluate(repaired, normal, start=3600, end=3900):
    selected = repaired.loc[repaired.Time.between(start, end)]
    merged = selected.merge(normal[['Time', 'Spacing', 'Speed_Follower']], on='Time',
                            validate='one_to_one', how='left', indicator=True)
    if not (merged['_merge'] == 'both').all():
        raise ValueError('Evaluation Time missing in normal reference')
    result = {'evaluation_rows': len(merged)}
    if merged.empty:
        return result
    for name, repaired_column, normal_column in [
        ('spacing', 'Repaired_Spacing', 'Spacing'),
        ('speed', 'Repaired_Speed', 'Speed_Follower')]:
        error = merged[repaired_column].to_numpy() - merged[normal_column].to_numpy()
        if not np.isfinite(error).all():
            raise ValueError('Nonfinite repair output')
        result[name+'_MAE'] = float(np.mean(np.abs(error)))
        result[name+'_RMSE'] = float(np.sqrt(np.mean(error**2)))
        result[name+'_MAPE_percent'] = float(np.mean(np.abs(error/(merged[normal_column].to_numpy()+1e-6)))*100)
    return result


def summarize(rows):
    frame = pd.DataFrame(rows)
    all_valid = len(frame) == 5 and (frame['status'] == 'valid').all()
    keys = [name+'_'+metric for name in ['spacing','speed']
            for metric in ['MAE','RMSE','MAPE_percent']]
    return pd.DataFrame([{'metric': c,
                          'mean': frame[c].mean() if all_valid else np.nan,
                          'std': frame[c].std(ddof=0) if all_valid else np.nan,
                          'valid_runs': int((frame.status == 'valid').sum()),
                          'status': 'valid' if all_valid else 'incomplete_five_runs'} for c in keys])

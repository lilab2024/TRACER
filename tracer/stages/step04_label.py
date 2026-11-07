import numpy as np


def label(frame, bounds):
    result = frame.copy()
    abnormal = np.zeros(len(frame), dtype=bool)
    for column in ['beta1', 'beta3']:
        low, high = bounds[column]
        outside = ~frame[column].between(low, high, inclusive='both')
        result[f'{column}_outside'] = outside.astype(int)
        abnormal |= outside.to_numpy()
    result['Label'] = np.where(abnormal, '异常', '正常')
    return result


def split_training(frame, attack_start=3600):
    expert = frame.loc[(frame.Time < attack_start) & (frame.Label == '正常')].copy()
    abnormal = frame.loc[frame.Label == '异常'].copy()
    if expert.empty:
        raise ValueError('No normal pre-attack training rows')
    return expert, abnormal

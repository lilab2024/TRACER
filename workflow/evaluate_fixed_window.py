"""Evaluate a fixed window: repair abnormal rows, retain other attack rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from stages.step06_metrics import evaluate, summarize


def run(result_dir, scenario, start=3600, steps=300):
    if steps <= 0:
        raise ValueError('steps must be positive')
    folder = result_dir/scenario
    output = folder/f'fixed_window_{start}_{start+steps-1}'
    output.mkdir(exist_ok=False)
    normal = pd.read_csv(result_dir/'normal/02_target_trimmed.txt', sep='\t')
    attack = pd.read_csv(folder/'04_target_labeled.txt', sep='\t')
    window = attack.loc[attack.Time.between(start, start+steps-1)].copy()
    assert window.Time.tolist() == list(range(start, start+steps)), 'Missing/duplicate window Time'
    expected_bad = set(attack.loc[attack.Label=='异常', 'Time'])
    rows=[]
    for iteration in range(1,6):
        repaired = pd.read_csv(folder/f'06_repaired_run_{iteration:02d}.txt',sep='\t')
        assert not repaired.Time.duplicated().any()
        assert set(repaired.Time)==expected_bad, 'Missing or unexpected repair rows'
        frame = window.merge(repaired[['Time','Repaired_Spacing','Repaired_Speed']],
                             on='Time', how='left', validate='one_to_one')
        bad = frame.Label=='异常'
        for repair_col, raw_col in [('Repaired_Spacing','Spacing'),('Repaired_Speed','Speed_Follower')]:
            frame[repair_col] = pd.to_numeric(frame[repair_col], errors='raise')
            assert np.isfinite(frame.loc[bad,repair_col]).all(), 'Invalid repaired values'
            frame.loc[~bad,repair_col] = frame.loc[~bad,raw_col]
        frame['Value_Source'] = np.where(bad,'repaired','original_attack')
        frame.to_csv(output/f'evaluated_trajectory_run_{iteration:02d}.txt',sep='\t',index=False,float_format='%.6f')
        values = evaluate(frame[['Time','Repaired_Spacing','Repaired_Speed']], normal, start=start, end=start+steps-1)
        assert values['evaluation_rows']==steps
        rows.append({'run':iteration, **values, 'status':'valid'})
    each=pd.DataFrame(rows);summary=summarize(rows)
    each.to_csv(output/'metrics_each_run.txt',sep='\t',index=False,float_format='%.6f')
    summary.to_csv(output/'metrics_mean_std.txt',sep='\t',index=False,float_format='%.6f')
    metadata={'scenario':scenario,'start_inclusive':start,'end_inclusive':start+steps-1,
              'steps':steps,'repaired_rows':int((window.Label=='异常').sum()),
              'original_attack_rows':int((window.Label=='正常').sum()),
              'training_repeated':False,'normal_reference':'normal/02_target_trimmed.txt'}
    (output/'evaluation_settings.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2));print(summary.to_string(index=False));print('Output:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir',type=Path)
    parser.add_argument('--scenario',required=True,choices=['ba','dpa','ma','pa','fa'])
    parser.add_argument('--start',type=int,default=3600)
    parser.add_argument('--steps',type=int,default=300)
    args=parser.parse_args()
    run(args.result_dir.resolve(),args.scenario,args.start,args.steps)

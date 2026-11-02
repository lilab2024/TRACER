import os
import pandas as pd
import numpy as np
trajectory_file = 'input_trajectory.txt'
output_file = 'beta.txt'
vehicle_params = {'v0': 39.12, 'tau': 0.96, 's0': 10.0, 'delta': 20.0, 'alpha': 1.18, 'beta': 8.9}
df_all = pd.read_csv(trajectory_file, delim_whitespace=True)
beta1_list = []
beta2_list = []
beta3_list = []
for (idx, row) in df_all.iterrows():
    v = row['Speed_Follower']
    delta_v = row['Relative_speed']
    s = row['Spacing']
    if s <= 1e-06:
        beta1_list.append(np.nan)
        beta2_list.append(np.nan)
        beta3_list.append(np.nan)
        continue
    s_star = vehicle_params['s0'] + v * vehicle_params['tau'] + v * delta_v / (2 * np.sqrt(vehicle_params['alpha'] * vehicle_params['beta']))
    beta1 = 2 * vehicle_params['alpha'] * s_star ** 2 / s ** 3
    beta2 = vehicle_params['alpha'] * (s_star * v) / (s ** 2 * np.sqrt(vehicle_params['alpha'] * vehicle_params['beta']))
    beta3 = -vehicle_params['alpha'] * (vehicle_params['delta'] * v ** (vehicle_params['delta'] - 1) / vehicle_params['v0'] ** vehicle_params['delta'] + 2 * s_star / s ** 2 * (vehicle_params['tau'] - delta_v / (2 * np.sqrt(vehicle_params['alpha'] * vehicle_params['beta']))))
    beta1_list.append(beta1)
    beta2_list.append(beta2)
    beta3_list.append(beta3)
df_all['beta1'] = beta1_list
df_all['beta2'] = beta2_list
df_all['beta3'] = beta3_list
with open(output_file, 'w') as f:
    f.write(f"{'Time':>6} {'Speed_Follower':>12} {'Relative_speed':>14} {'Spacing':>12} {'beta1':>12} {'beta2':>12} {'beta3':>12}\n")
    for (idx, row) in df_all.iterrows():
        f.write(f"{row['Time']:>6} {row['Speed_Follower']:>12.6f} {row['Relative_speed']:>14.6f} {row['Spacing']:>12.6f} {row['beta1']:>12.6f} {row['beta2']:>12.6f} {row['beta3']:>12.6f}\n")
print(f'Data with beta1, beta2 and beta3 saved to {output_file}')
beta1_min = np.nanmin(df_all['beta1'])
beta1_max = np.nanmax(df_all['beta1'])
beta2_min = np.nanmin(df_all['beta2'])
beta2_max = np.nanmax(df_all['beta2'])
beta3_min = np.nanmin(df_all['beta3'])
beta3_max = np.nanmax(df_all['beta3'])
print('\n========== Beta Range ==========')
print(f'beta1: Min = {beta1_min:.6f}, Max = {beta1_max:.6f}')
print(f'beta2: Min = {beta2_min:.6f}, Max = {beta2_max:.6f}')
print(f'beta3: Min = {beta3_min:.6f}, Max = {beta3_max:.6f}')
print('================================')

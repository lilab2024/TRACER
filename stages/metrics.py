import pandas as pd
import numpy as np
normal_file = 'normal_beta.txt'
repaired_files = ['offline_lstm_repaired_output_Run1.txt', 'offline_lstm_repaired_output_Run2.txt', 'offline_lstm_repaired_output_Run3.txt', 'offline_lstm_repaired_output_Run4.txt', 'offline_lstm_repaired_output_Run5.txt']
df_normal = pd.read_csv(normal_file, sep='\\s+', engine='python')
df_normal.columns = df_normal.columns.str.strip()

def compute_metrics(df):
    epsilon = 1e-06
    speed_error = df['Repaired_Speed'] - df['Speed_Follower']
    speed_mae = np.mean(np.abs(speed_error))
    speed_rmse = np.sqrt(np.mean(speed_error ** 2))
    speed_mape = np.mean(np.abs(speed_error / (df['Speed_Follower'] + epsilon))) * 100
    spacing_error = df['Repaired_Spacing'] - df['Spacing']
    spacing_mae = np.mean(np.abs(spacing_error))
    spacing_rmse = np.sqrt(np.mean(spacing_error ** 2))
    spacing_mape = np.mean(np.abs(spacing_error / (df['Spacing'] + epsilon))) * 100
    return {'speed_mae': speed_mae, 'speed_rmse': speed_rmse, 'speed_mape': speed_mape, 'spacing_mae': spacing_mae, 'spacing_rmse': spacing_rmse, 'spacing_mape': spacing_mape}
(speed_mae_list, speed_rmse_list, speed_mape_list) = ([], [], [])
(spacing_mae_list, spacing_rmse_list, spacing_mape_list) = ([], [], [])
for file in repaired_files:
    df_rep = pd.read_csv(file, sep='\\s+', engine='python')
    df_rep.columns = df_rep.columns.str.strip()
    df = pd.merge(df_rep, df_normal[['Time', 'Speed_Follower', 'Spacing']], on='Time', how='inner')
    df = df[(df['Time'] >= 3600) & (df['Time'] <= 3900)].copy()
    if df.empty:
        print(f'{file}: 无有效数据')
        continue
    metrics = compute_metrics(df)
    speed_mae_list.append(metrics['speed_mae'])
    speed_rmse_list.append(metrics['speed_rmse'])
    speed_mape_list.append(metrics['speed_mape'])
    spacing_mae_list.append(metrics['spacing_mae'])
    spacing_rmse_list.append(metrics['spacing_rmse'])
    spacing_mape_list.append(metrics['spacing_mape'])

def mean_std(x):
    return (np.mean(x), np.std(x))
(speed_mae_mean, speed_mae_std) = mean_std(speed_mae_list)
(speed_rmse_mean, speed_rmse_std) = mean_std(speed_rmse_list)
(speed_mape_mean, speed_mape_std) = mean_std(speed_mape_list)
(spacing_mae_mean, spacing_mae_std) = mean_std(spacing_mae_list)
(spacing_rmse_mean, spacing_rmse_std) = mean_std(spacing_rmse_list)
(spacing_mape_mean, spacing_mape_std) = mean_std(spacing_mape_list)
print('\n================ SPEED =================')
print(f'MAE  : {speed_mae_mean:.6f} ± {speed_mae_std:.6f}')
print(f'RMSE : {speed_rmse_mean:.6f} ± {speed_rmse_std:.6f}')
print(f'MAPE : {speed_mape_mean:.4f}% ± {speed_mape_std:.4f}%')
print('\n================ SPACING ===============')
print(f'MAE  : {spacing_mae_mean:.6f} ± {spacing_mae_std:.6f}')
print(f'RMSE : {spacing_rmse_mean:.6f} ± {spacing_rmse_std:.6f}')
print(f'MAPE : {spacing_mape_mean:.4f}% ± {spacing_mape_std:.4f}%')
print('\n完成：五次实验统计（mean ± std）已输出')

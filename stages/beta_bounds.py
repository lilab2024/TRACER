import pandas as pd
file_path = 'normal_beta.txt'
df = pd.read_csv(file_path, delim_whitespace=True)
df_last_2000 = df.iloc[2000:]
max_beta1 = df_last_2000['beta1'].max()
min_beta1 = df_last_2000['beta1'].min()
max_beta2 = df_last_2000['beta2'].max()
min_beta2 = df_last_2000['beta2'].min()
max_beta3 = df_last_2000['beta3'].max()
min_beta3 = df_last_2000['beta3'].min()
print(f'最大 beta1: {max_beta1}')
print(f'最小 beta1: {min_beta1}')
print(f'最大 beta2: {max_beta2}')
print(f'最小 beta2: {min_beta2}')
print(f'最大 beta3: {max_beta3}')
print(f'最小 beta3: {min_beta3}')

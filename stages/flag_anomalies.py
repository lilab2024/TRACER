import pandas as pd
file_path = 'attack_beta.txt'
output_path = 'attack_beta_flag.txt'
(beta1_min, beta1_max) = (0.00259, 0.07883)
(beta3_min, beta3_max) = (-0.071054, -0.011188)
df = pd.read_csv(file_path, delim_whitespace=True)
df['flag1'] = df['beta1'].apply(lambda x: '正常' if beta1_min <= x <= beta1_max else '异常')
df['flag2'] = df['beta3'].apply(lambda x: '正常' if beta3_min <= x <= beta3_max else '异常')
col_widths = []
for col in df.columns:
    max_len = max(df[col].astype(str).apply(len).max(), len(col))
    col_widths.append(max_len + 2)
header = ''.join((col.ljust(w) for (col, w) in zip(df.columns, col_widths)))
rows = []
for (_, row) in df.iterrows():
    line = ''.join((str(row[col]).ljust(w) for (col, w) in zip(df.columns, col_widths)))
    rows.append(line)
output_text = header + '\n' + '\n'.join(rows)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(output_text)
print(output_text)
print(f'\n✅ 已保存严格对齐后的文件：{output_path}')

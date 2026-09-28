import pandas as pd
import os

csv_dir = 'csvs/'

dfs = {f: pd.read_csv(csv_dir+f) for f in os.listdir(csv_dir)}

for k, df in dfs.items():
	print(k)
	all_trigger_names = df.columns[3:-2].tolist()
	trigger_names = all_trigger_names[:len(all_trigger_names)//2]
	excl_trigger_names = all_trigger_names[len(all_trigger_names)//2:]
	if len(trigger_names) == 2:
		print('2 triggers: onlyA + onlyB + AandB = AorB?', (df[[*excl_trigger_names, 'all']].sum(axis=1) == df['any']).all())
	for col in df.columns[3:]:
			df[f"eff_{col}"] = df[col]/df['tot_n_evts']
	print(df.columns)
	print(df.to_string(header=False))
	print('max eff_any: ', df[df['eff_any']==df['eff_any'].max()].to_string(header=False))
	for trg_name in trigger_names:
		print(f'max eff_{trg_name}: ', df[df[f'eff_{trg_name}']==df[f'eff_{trg_name}'].max()].to_string(header=False))
	print()

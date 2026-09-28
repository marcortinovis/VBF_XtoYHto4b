import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


with open("../../var_labels.json", "r") as file:
    var_labels = json.load(file)

parser = argparse.ArgumentParser(description="Dataset preparation for the DNN-based VBF Jet identification.")
parser.add_argument("--MX", type=str, default=None, help="Mass of the X particle (MX).")
parser.add_argument("--MY", type=str, default=None, help="Mass of the Y particle (MY).")
parser.add_argument("--Mall", action="store_true", default=None, help="Run the training for all the mass combinations.")
parser.add_argument("--d_id", type=str, required=True, help="The dataset type id (alpha or beta).")
parser.add_argument
#parser.add_argument("--debug", action="store_true", help="Debug: runs script on just a few events.")
args = parser.parse_args()

if args.d_id!='beta':
    if args.Mall is None:
        if args.MX is None or args.MY is None:
            parser.error("Either --Mall or both --MX and --MY must be provided")
    elif args.MX is not None or args.MY is not None:
        parser.error("--Mall cannot be used with --MX or --MY")


m_combs = ['MX_1000_MY_125',
		 'MX_500_MY_125', 'MX_700_MY_125', 'MX_1400_MY_125', 'MX_2000_MY_125',
		 'MX_1000_MY_90', 'MX_1000_MY_200', 'MX_1000_MY_400', 'MX_1000_MY_600']
if args.Mall is None and args.d_id!='beta':
    m_combs = [m_comb for m_comb in m_combs if (args.MX in m_comb and args.MY in m_comb)]


orig_path = '../../mjj_sel/vbf_parquets/'


vars_to_log = {'jet_props_1_pT': 15., 'jet_props_2_pT': 15., 'mjj': 0., 'pTjj': 0.}
vars_to_abs = ['ang_seps_1_DeltaRap_0', 'ang_seps_1_DeltaRap_1', 'ang_seps_1_DeltaRap_2', 'ang_seps_1_DeltaRap_3',
               'ang_seps_2_DeltaRap_0', 'ang_seps_2_DeltaRap_1', 'ang_seps_2_DeltaRap_2', 'ang_seps_2_DeltaRap_3',
               'dEta', 'jet_props_1_rap', 'jet_props_2_rap']
vars_to_excl = [
    'jet_iso_1_neighbour_id', 'jet_iso_2_neighbour_id', 
    'jet_props_1_JetbtagUParTAK4SvUDG', 'jet_props_2_JetbtagUParTAK4SvUDG',
    'cos_theta'
]

out_dir_root = './'
os.makedirs(out_dir_root, exist_ok=True)

# Load
dfs = {m_comb: pd.read_parquet(orig_path+'vars_'+m_comb+'.parquet').drop(columns=vars_to_excl) for m_comb in m_combs}
lens = [len(df) for df in dfs.values()]
if args.d_id == 'alpha':
    True
elif args.d_id == 'beta':
    dfs = {'Mall': pd.concat(dfs.values())}
    print(len(next(iter(dfs.values()))), sum(lens))


for k, df in dfs.items():

    print(f'Processing {k}')

    lim = 6


    # Transform

    for var, cut in vars_to_log.items():
        df[var] = np.log(df[var] - cut + 1)
    for var in vars_to_abs:
        df[var] = np.abs(df[var])


    # Balance

    nSig = (df['is_vbf'] == 1).sum()
    nBkg = (df['is_vbf'] == 0).sum()
    nMin = min(nSig, nBkg)
    print('nSig', nSig, 'nBkg', nBkg)
    if nSig+nBkg != len(df):
        print('Error: nSim + nBkg != len of the dataset')
        sys.exit(1)
    sig_pos = np.where(df['is_vbf'].to_numpy() == 1)[0]
    bkg_pos = np.where(df['is_vbf'].to_numpy() == 0)[0]
    sig_pos = np.random.choice(sig_pos, size=nMin, replace=False)
    bkg_pos = np.random.choice(bkg_pos, size=nMin, replace=False)
    pos = np.concatenate([sig_pos, bkg_pos])
    df = df.iloc[pos]


    # Standardization    

    df = df.replace([np.inf, -np.inf], np.nan)

    df_trainval, df_test = train_test_split(df, test_size=0.1, random_state=125)
    print('trainval', len(df_trainval), 'test', len(df_test))

    trainval_mean = df_trainval[df.columns[lim:]].mean()
    trainval_std = df_trainval[df.columns[lim:]].std()
    df_trainval_norm = df_trainval.copy()
    df_trainval_norm[df.columns[lim:]] = (df_trainval[df.columns[lim:]] - trainval_mean) / trainval_std
    df_train_norm, df_val_norm = train_test_split(df_trainval_norm, test_size=1./9, random_state=125)
    print('train', len(df_train_norm), 'val', len(df_val_norm), 'test', len(df_test))

    test_mean = df_test[df.columns[lim:]].mean()
    test_std = df_test[df.columns[lim:]].std()
    df_test_norm = df_test.copy()
    df_test_norm[df.columns[lim:]] = (df_test[df.columns[lim:]] - test_mean) / test_std

    out_dir = out_dir_root+f'datasets_{args.d_id}/dataset_{args.d_id}_'+k+'/'
    os.makedirs(out_dir, exist_ok=True)

    print(df_train_norm.iloc[0].to_dict())


    # Saving

    df_train_norm.to_parquet(out_dir+k+'_train_set.parquet', index=False)
    df_val_norm.to_parquet(out_dir+k+'_val_set.parquet', index=False)
    df_test_norm.to_parquet(out_dir+k+'_test_set.parquet', index=False)
    with open(out_dir+k+'_trainval_mean_std.json', "w") as f:
        json.dump({
            'mean': {feature: trainval_mean[feature] for feature in trainval_mean.index},
            'std': {feature: trainval_std[feature] for feature in trainval_std.index}
        }, f)
    with open(out_dir+k+'_used_vars.txt', "w", encoding="utf-8") as f:
        for col in df_test_norm.columns[lim:]:
            f.write(f"{col}\n")
    with open(out_dir+k+'_log_vars.json', "w") as f:
        json.dump(vars_to_log, f)
    with open(out_dir+k+'_abs_vars.txt', "w", encoding="utf-8") as f:
        for v in vars_to_abs:
            f.write(f"{v}\n")
    print(f'{k} sets saved')


    # Correlation

    corr_mtrx = df_train_norm[df.columns[lim:]].corr()

    corr_mtrx = corr_mtrx.rename(index=var_labels, columns=var_labels)

    corr_no_diag = corr_mtrx.copy()
    np.fill_diagonal(corr_no_diag.values, np.nan)
    perfect_corr = corr_no_diag[(corr_no_diag==1) | (corr_no_diag==-1)]
    if perfect_corr.notna().any().any():
        print(perfect_corr)
    
    fig, ax = plt.subplots(figsize=(14,12))
    m_comb_title = rf"$M_X={k.split('_')[1]}$ GeV, $M_Y={k.split('_')[3]}$ GeV"
    ax.set_title(f'Correlation of scaled and normalized features\n Mass combination {m_comb_title}')
    sns.heatmap(corr_mtrx, cmap='coolwarm', annot=False, center=0, vmin=-1, vmax=1)
    plt.savefig(out_dir+k+'_correlation_heatmap.png', dpi=300, bbox_inches="tight")
    plt.savefig(out_dir+k+'_correlation_heatmap.pdf', dpi=300, bbox_inches="tight")
    plt.close()

    print()
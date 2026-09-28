import os
from pathlib import Path
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import argparse
import mplhep as mh
mh.style.use('CMS')

frmts = ['.png', '.pdf']


parser = argparse.ArgumentParser(description="Make cut flow plots.")
parser.add_argument("--mjj", action="store_true", help="Choose mjj-based selection of the VBF Jet pair.")
parser.add_argument("--dnn", action="store_true", help="Choose DNN-based selection of the VBF Jet pair.")
parser.add_argument("--fancy", action="store_true", help="Fancy plotting.")
args = parser.parse_args()

if sum([args.mjj, args.dnn]) != 1:
	parser.error("You must specify either --mjj or --dnn.")
if args.dnn:
	root_dir = 'dnn_sel/'
elif args.mjj:
	root_dir = 'mjj_sel/'


with open("var_labels.json", "r") as file:
    var_labels = json.load(file)

in_dir = root_dir+'vbf_parquets/'
parquet_fnames = [in_dir+p for p in os.listdir(in_dir)]

out_root_dir = root_dir+'plots/vars_plots/'
os.makedirs(out_root_dir, exist_ok=True)

# Define custom parameters to plot specific features
custom_params = {
                    'def':                                  {'xmin': np.nan,    'xmax': np.nan,     'xscale': 'linear',     'yscale': 'linear'},
                    'zepp':                                 {'xmin': -7.5,      'xmax': 7.5,        'xscale': 'linear',     'yscale': 'log'},
                    'jet_props_1_JetbtagUParTAK4SvUDG':     {'xmin': 1e-3,      'xmax': 1,          'xscale': 'log',        'yscale': 'linear'}, 
                    'jet_props_1_btagUParTAK4B':            {'xmin': 1e-4,      'xmax': 1,          'xscale': 'log',        'yscale': 'linear'}, 
                    'jet_props_1_btagUParTAK4QvG':          {'xmin': 0,         'xmax': 1,          'xscale': 'linear',     'yscale': 'linear'},
                    'jet_props_2_JetbtagUParTAK4SvUDG':     {'xmin': 5e-3,      'xmax': 1,          'xscale': 'log',        'yscale': 'linear'}, 
                    'jet_props_2_btagUParTAK4B':            {'xmin': 1e-4,      'xmax': 1,          'xscale': 'log',        'yscale': 'linear'}, 
                    'jet_props_2_btagUParTAK4QvG':          {'xmin': 0,         'xmax': 1,          'xscale': 'linear',     'yscale': 'linear'},
                    'jet_props_1_pT':                       {'xmin': 10,        'xmax': 500,        'xscale': 'log',        'yscale': 'linear'},
                    'jet_props_2_pT':                       {'xmin': 10,        'xmax': 500,        'xscale': 'log',        'yscale': 'linear'},
                    'pTjj':                                 {'xmin': 5,         'xmax': 1000,       'xscale': 'log',        'yscale': 'linear'},
                    'mjj':                                  {'xmin': 5,         'xmax': 8000,       'xscale': 'log',        'yscale': 'linear'},
                }

# Cycle between samples
for parquet_fname in parquet_fnames:
    print(parquet_fname)
    parq = pd.read_parquet(parquet_fname)
    out_dir = out_root_dir+str(Path(parquet_fname).stem)+'/'
    os.makedirs(out_dir, exist_ok=True)
    sig = parq[parq['is_vbf'] == 1]
    bkg = parq[parq['is_vbf'] == 0]
    cols = parq.columns
    fig, ax = plt.subplots()
    for col in cols[6:]:
        custom_col = True if col in custom_params.keys() else False
        params = custom_params[col] if custom_col else custom_params['def']
        ax.clear()
        print(col)
        print('s', 'b')
        s = sig[col]
        b = bkg[col]
        print(len(s), len(b))
        s = s[
            np.isfinite(s)
            & (s < (10e6 if not np.isfinite(params['xmax']) else params['xmax']))
            & (s > (-10e6 if not np.isfinite(params['xmin']) else params['xmin']))
        ]
        b = b[
            np.isfinite(b) 
            & (b < (10e6 if not np.isfinite(params['xmax']) else params['xmax']))
            & (b > (-10e6 if not np.isfinite(params['xmin']) else params['xmin']))
        ]
        print(len(s), len(b))
        if params['xscale'] == 'log':
            edges = np.logspace(np.log10(min(s.min(), b.min())), np.log10(max(s.max(), b.max())), 100 + 1)
        else:
            edges = np.linspace(min(s.min(), b.min()), max(s.max(), b.max()), 100 + 1) 
        ax.hist(s, bins=edges, density=True, histtype='step', label='SIG'+('' if args.fancy else f' (VBF jets correct match) ({(len(s)/(len(s)+len(b))):.2f})'))
        ax.hist(b, bins=edges, density=True, histtype='step', label='BKG'+('' if args.fancy else f' (VBF jets mismatch) ({(len(b)/(len(s)+len(b))):.2f})'))
        if np.isfinite(params['xmin']) and np.isfinite(params['xmax']):
            ax.set_xlim(params['xmin'], params['xmax'])
        ax.set_xscale(params['xscale'])
        ax.set_yscale(params['yscale'])
        ax.legend()
        ax.set_xlabel(var_labels[col]+(' (log)' if params['xscale']=='log' else ''))
        ax.set_ylabel('Density (normalized)'+(' (log)' if params['yscale']=='log' else ''))
        mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
        for frmt in frmts:
            fig.savefig(out_dir+col+'.pdf')
            print(out_dir+col+'.pdf', 'saved')
        print()
    plt.close()
import os
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import mplhep as mh
mh.style.use('CMS')


parser = argparse.ArgumentParser(description="Make bjet plots.")
parser.add_argument("--mjj", action="store_true", help="Choose mjj-based selection of the VBF Jet pair.")
parser.add_argument("--dnn", action="store_true", help="Choose DNN-based selection of the VBF Jet pair.")
args = parser.parse_args()

if sum([args.mjj, args.dnn]) != 1:
	parser.error("You must specify either --mjj or --dnn.")
if args.dnn:
	root_dir = 'dnn_sel/'
elif args.mjj:
	root_dir = 'mjj_sel/'

var_labels = {
      'pt_0': r'$p_T^{(0)}$', 'pt_1': r'$p_T^{(1)}$', 'pt_2': r'$p_T^{(2)}$', 'pt_3': r'$p_T^{(3)}$',
      'eta_0': r'$\eta^{(0)}$', 'eta_1': r'$\eta^{(1)}$', 'eta_2': r'$\eta^{(2)}$', 'eta_3': r'$\eta^{(3)}$', 
      'phi_0': r'$\phi^{(0)}$', 'phi_1': r'$\phi^{(1)}$', 'phi_2': r'$\phi^{(2)}$', 'phi_3': r'$\phi^{(3)}$', 
      'm_0': r'$m^{(0)}$', 'm_1': r'$m^{(1)}$', 'm_2': r'$m^{(2)}$', 'm_3': r'$m^{(3)}$',
      'pt_X': r'$p_T^{(4b)}$', 'eta_X': r'$\eta^{(4b)}$', 'phi_X': r'$\phi^{(4b)}$', 'm_X': r'$m^{(4b)}$',
      'score_0': r'$btagUParTAK4B^{(0)}$', 'score_1': r'$btagUParTAK4B^{(1)}$', 'score_2': r'$btagUParTAK4B^{(2)}$', 'score_3': r'$btagUParTAK4B^{(3)}$'
      }

in_dir = root_dir+'bjets/'
parquet_fnames = [in_dir+p for p in os.listdir(in_dir)]

out_root_dir = root_dir+'plots/bjets_plots/'
os.makedirs(out_root_dir, exist_ok=True)


for parquet_fname in parquet_fnames:
    print(parquet_fname)
    parq = pd.read_parquet(parquet_fname)
    out_dir = out_root_dir+str(Path(parquet_fname).stem)+'/'
    os.makedirs(out_dir, exist_ok=True)
    parq_pass = parq[parq['passed']]
    cols = parq.columns
    for col in cols[3:]:
        fig, ax = plt.subplots()
        p = parq_pass[col]
        p = np.asarray(p, dtype=float)
        p = p[np.isfinite(p)]
        edges = np.linspace(p.min(), p.max(), 100 + 1) 
        ax.hist(p, bins=edges, density=True, histtype='step')
        ax.set_xlabel(var_labels[col]+(' [GeV]' if any(x in col for x in ['m', 'pt']) else ''))
        mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
        figname = out_dir+col+'.png'
        fig.savefig(figname, bbox_inches='tight')
        print(figname, 'saved')
        ax.clear()
        plt.close(fig)
        if any(x in col for x in ['m', 'pt']):
            fig, ax = plt.subplots()
            edges = np.logspace(np.log10(p.min()), np.log10(p.max()), 100 + 1)
            ax.hist(p, bins=edges, density=True, histtype='step')
            ax.set_xlabel(var_labels[col]+' (log)')
            ax.set_xscale('log')
            mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
            figname = out_dir+'log_'+col+'.png'
            fig.savefig(figname, bbox_inches='tight')
            print(figname, 'saved')
            ax.clear()
            plt.close(fig)

    plt.close()
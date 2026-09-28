from matplotlib import pyplot as plt
import matplotlib as mpl
custom_colours = ["indianred", "cornflowerblue", "darkseagreen", "peru", "lightpink", "khaki", "mediumpurple", "grey"]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colours)
import pandas as pd
import numpy as np
import os
import argparse
import mplhep as mh
mh.style.use('CMS')

frmts = ['.pdf', '.png']


parser = argparse.ArgumentParser(description="Make cut flow plots.")
parser.add_argument("--mjj", action="store_true", help="Choose mjj-based selection of the VBF Jet pair.")
parser.add_argument("--dnn", action="store_true", help="Choose DNN-based selection of the VBF Jet pair.")
args = parser.parse_args()

if sum([args.mjj, args.dnn]) != 1:
	parser.error("You must specify either --mjj or --dnn.")
if args.dnn:
	root_dir = 'dnn_sel/'
elif args.mjj:
	root_dir = 'mjj_sel/'


plot_folder = root_dir+'plots/cut_flow_plots/'
os.makedirs(plot_folder, exist_ok=True)

m_combs = ['MX_1000_MY_125',
		 'MX_500_MY_125', 'MX_700_MY_125', 'MX_1400_MY_125', 'MX_2000_MY_125',
		 'MX_1000_MY_90', 'MX_1000_MY_200', 'MX_1000_MY_400', 'MX_1000_MY_600']

# Dataframes loading
df_QCD = pd.read_csv(root_dir+'csvs/QCD.csv')
df_VBF = pd.read_csv(root_dir+'csvs/VBF.csv')
df_ggF = pd.read_csv(root_dir+'csvs/ggF.csv')

cols = ['No cuts', 'Trigger', '# of jets\n>= 6', 'Loose b\nWP', 'VBF Cuts']

# Data collection
data = {}
row_QCD = df_QCD[df_QCD['sample'] == 'QCD'].values.tolist()[0]
row_QCD = [r / row_QCD[1] for r in row_QCD[1:]]
row_QCD = row_QCD[:-2]
data['QCD'] = row_QCD
for comb in m_combs:
    row_ggF = df_ggF[df_ggF['sample'] == comb].values.tolist()[0]
    row_ggF = [r / row_ggF[1] for r in row_ggF[1:]]
    row_ggF = row_ggF[:-2]
    data['ggF_'+comb] = row_ggF
for comb in m_combs:
    row_VBF = df_VBF[df_VBF['sample'] == comb].values.tolist()[0]
    row_VBF = [r / row_VBF[1] for r in row_VBF[1:]]
    row_VBF = row_VBF[:-2]
    data['VBF_'+comb] = row_VBF

print(data)



def make_stairplot(k, v, fig = None, title=None, fnamelab=None, ylog=False):
    """
    Make the cut-flow plots
    """
    if fig is None:
        fig, ax = plt.subplots(layout='constrained')
    else:
        ax = fig.axes[0]
    x = np.arange(len(v))
    edges = np.arange(x[0]-0.5, x[-1]+0.51, 1.)
    splits = k.split("_",1)
    lab = '' if splits[0]=='QCD' else rf'($M_X={int(splits[-1].split("_")[1]):4d}$ GeV, $M_Y={int(splits[-1].split("_")[3]):4d}$ GeV)'
    if splits[0]=='QCD':
        ax.stairs(v, edges, label=lab, lw=2, color='darkseagreen')
    else:
        ax.stairs(v, edges, label=lab, lw=2)
    if ylog:
        ax.set_yscale('log')
    ax.set_xticks(x, cols)
    ax.set_ylabel('Fraction'+(' (log)' if ylog else ''))
    if title is None:
        title = k
    ax.text(
        0.8-0.075, 0.8-0.025, k.split("_",1)[0],
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=36,
    )
    ax.legend(loc="lower left", fontsize=18, bbox_to_anchor=(0.05, 0.025))
    mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
    for frmt in frmts:
        fig.savefig(plot_folder+f'cut_flow_{fnamelab if fnamelab is not None else "XYZ"}'+frmt, bbox_inches='tight')
    return fig

# Make individual (specific per mass point) cut-flow plots
for k, v in data.items():
    make_stairplot(k, v, fnamelab=k, ylog=(True if k=='QCD' else False))

vbf_yscan = {k: v for k, v in data.items() if 'MX_1000' in k and 'VBF' in k}
vbf_xscan = {k: v for k, v in data.items() if 'MY_125' in k and 'VBF' in k}
ggf_yscan = {k: v for k, v in data.items() if 'MX_1000' in k and 'ggF' in k}
ggf_xscan = {k: v for k, v in data.items() if 'MY_125' in k and 'ggF' in k}

# Make cut-flow plots for the MX/MY scans
fig = None
for k, v in vbf_xscan.items():
    fig = make_stairplot(k, v, fig=fig, title='Cut Flow - VBF MX Scan', fnamelab='VBF_MX_scan')
plt.close(fig)
fig = None
for k, v in vbf_yscan.items():
    fig = make_stairplot(k, v, fig=fig, title='Cut Flow - VBF MY Scan', fnamelab='VBF_MY_scan')
plt.close(fig)
fig = None
for k, v in ggf_xscan.items():
    fig = make_stairplot(k, v, fig=fig, title='Cut Flow - ggF MX Scan', fnamelab='ggF_MX_scan')
plt.close(fig)
fig = None
for k, v in ggf_yscan.items():
    fig = make_stairplot(k, v, fig=fig, title='Cut Flow - ggF MY Scan', fnamelab='ggF_MY_scan')
plt.close(fig)


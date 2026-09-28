import matplotlib as mpl
custom_colours = ["indianred", "cornflowerblue", "darkseagreen", "peru", "lightpink", "khaki", "mediumpurple", "grey"]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colours)
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path
import shap
import gc
from sklearn.metrics import roc_curve, auc
import mplhep as mh
mh.style.use('CMS')

frmts = ['.png', '.pdf']

with open('../var_labels.json') as f:
	feats_dict = json.load(f)


def plot_running_metric(df, name, outdir):
	epochs = df['epoch'].to_numpy()
	cols = [col for col in df.columns if name in col]
	values = {col: df[col].to_numpy() for col in cols}
	fig, ax = plt.subplots(figsize=(10, 8))
	for col, val in values.items():
		ax.plot(epochs, val, label=col.replace('_', ' ').capitalize())
	ax.grid()
	ax.set_xlabel('Epoch')
	ax.set_ylabel(name.capitalize())
	ax.legend()
	for frmt in frmts:
		figname = outdir+name+frmt
		fig.savefig(outdir+name+frmt, bbox_inches='tight')
		print(f'{figname} saved')
	plt.close(fig)


def plot_roc_curve(y_pred, y_true, outdir):

	fpr, tpr, thresholds = roc_curve(y_true, y_pred)
	roc_auc = auc(fpr, tpr)

	fig, ax = plt.subplots()
	ax.plot(fpr, tpr, lw=3, label=f'AUC = {roc_auc:.2f}')
	ax.plot([0, 1], [0, 1], linestyle='dashed', color='gray')
	xy = (0.15, 0.85)
	xytext = (0.45, 0.55)

	plt.annotate(
		'',
		xy=xy,
		xytext=xytext,
		arrowprops=dict(
			arrowstyle='simple',
			color='indianred',
			lw=2,
			ec=(0.5, 0., 0., 0.1),
			fc=(0.5, 0., 0., 0.1)
		)
	)
	midpoint = (
		0.075+(xy[0] + xytext[0]) / 2,
		0.025+(xy[1] + xytext[1]) / 2
	)
	plt.annotate(
		'Better',
		xy=midpoint,
		ha='center',
		va='center',
		#fontsize=20,
    	alpha=0.7
	)
	ax.legend()
	ax.set_xlabel('FPR')
	ax.set_ylabel('TPR')
	ax.legend(loc='lower right')
	ax.set_aspect('equal')
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.grid()
	for frmt in frmts:
		figname = outdir+'roc_curve'+frmt
		fig.savefig(figname, bbox_inches='tight')
		print(f'{figname} saved')
	plt.close(fig)


	fig, ax = plt.subplots()
	ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
	ax.legend()
	ax.set_xlabel('FPR')
	ax.set_ylabel('TPR')
	ax.legend(loc='lower right')
	ax.set_aspect('equal')
	ax.set_xlim(1.e-4, 1)
	ax.set_ylim(1.e-4, 1)
	ax.set_xscale('log') # requires xlim[0] > 0 (slightly)
	ax.set_yscale('log')
	ax.grid(which='both')
	for frmt in frmts:
		figname = outdir+'roc_curve_log'+frmt
		fig.savefig(figname, bbox_inches='tight')
		print(f'{figname} saved')
	plt.close(fig)


def plot_scores(train_pred, train_true, test_pred, test_true, outdir):
	fig, ax = plt.subplots(figsize=(10, 8))
	bins = np.linspace(0., 1., 50+1)
	ax.hist(train_pred[np.where(train_true==1)], bins=bins, density=True, label='Signal training set')
	ax.hist(train_pred[np.where(train_true==0)], bins=bins, density=True, label='Background training set')
	ax.hist(test_pred[np.where(test_true==1)], bins=bins, density=True, histtype='step', label='Signal test set')
	ax.hist(test_pred[np.where(test_true==0)], bins=bins, density=True, histtype='step', label='Background test set')
	ax.set_xlabel('Score')
	ax.legend()
	for frmt in frmts:
		figname = outdir+'scores'+frmt
		fig.savefig(figname, bbox_inches='tight')
		print(f'{figname} saved')
	plt.close(fig)

def compute_purity(test_pred, test_true, outdir):
	preds = [round(p) for p in test_pred]
	pure = sum(p == t for p, t in zip(preds, test_true))
	purity = float(pure)/len(test_true)
	if outdir is not None:
		purity_file = outdir + "purity.txt"
		with open(purity_file, "w", encoding="utf-8") as f:
			f.write("Purity\n\n")
			f.write(f"{pure}/{len(test_true)}\n")
			f.write(f"=\n")
			f.write(f"{purity}\n")
	return purity

def plot_purity_2d_scan(d, outdir):
	purity_dict = {k: compute_purity(test_pred=v['test']['y_pred'], test_true=v['test']['y_true'], outdir=None) for k, v in d.items()}
	rows = []
	for key, purity in purity_dict.items():
		match = re.search(r'MX_(\d+)_MY_(\d+)', key)
		if match:
			mx, my = map(int, match.groups())
			rows.append({'MX': mx, 'MY': my, 'purity': purity})
	df = pd.DataFrame(rows)
	fig, ax = plt.subplots()
	ax.set_axisbelow(True)
	ax.grid()
	scatter = ax.scatter(df['MX'], df['MY'], c=df['purity'], cmap='turbo', vmin=0, vmax=1)
	cbar = fig.colorbar(scatter)
	cbar.set_ticks(np.arange(0., 1.+0.1, 0.1))
	for i, p in df.iterrows():
		ax.annotate(
			f"{p['purity']:.3f}",
			(p["MX"], p["MY"]),
			textcoords="offset points",
			xytext=(5, 5),
			fontsize=20,
		)
	ax.set_xlabel("MX")
	ax.set_ylabel("MY")
	id = re.search(r'\d{3}', outdir).group()
	ax.set_title(f"Purity scan of the DNN VBF Jet selection\n {id}")
	for frmt in frmts:
		figname = outdir+f"{outdir.split('/')[-2]}_purity2Dscan"+frmt
		fig.savefig(figname, bbox_inches='tight', dpi=500)
		print(figname, 'saved')


def plot_shap(id=None):

	if id is not None:
		files = [
			path for path in Path(f"results/results_{id}").rglob("*.npz")
			if "shap_values" in path.name
		]
	else:
		files = [
			path for path in Path(f"results/").rglob("*.npz")
			if "shap_values" in path.name
		]

	for file in files:
		print(file)

		out_dir = str(file.parent)+'/'

		saved = np.load(file)
		with open(str(next(Path(out_dir).glob("*.txt"))), "r") as f:
			feats = f.read().splitlines()

		print('file loaded')

		shap_values = saved["shap_values"]
		X = saved["data"]
		feature_names = [feats_dict[f] for f in feats]
		print(feature_names)

		shap_expl = shap.Explanation(
			values=shap_values,
			data=X,
			feature_names=feature_names
		)

		print('exlanation loaded')

		
		importance = np.abs(shap_values).mean(axis=0)
		order = np.argsort(importance)[::-1]
		top9 = order[:9]
		others = order[9:]
		labels = [feature_names[i] for i in top9][::-1]
		values = importance[top9][::-1]
		fig, ax = plt.subplots()
		bars = ax.barh(labels, values)
		ax.set_xlabel("Mean |SHAP value|")
		for bar, value in zip(bars, values):
			ax.text(
				bar.get_width() - 0.03 * values.max(),
				bar.get_y() + bar.get_height() / 2,
				f"{value:.3f}",
				ha="right",
				va="center",
				color="white",
				fontweight="bold"
			)
		ax.set_xlim(0, values.max() * 1.15)
		plt.tight_layout()
		for frmt in frmts:
			plt.savefig(out_dir + "shap_feature_importance"+frmt, dpi=150, bbox_inches="tight")
		plt.close()


		fig, ax = plt.subplots()
		shap.plots.beeswarm(shap_expl, show=False)
		ax.set_yticklabels(['Other features']+[f'{labels[i]} ({values[i]:.3f})' for i in range(len(labels))])
		ax.set_xlabel("SHAP value")
		#--
		ax.text(
		    -0.17, 0.95,
		    "Feature name\n(Feature importance)",
		    transform=ax.transAxes,
		    ha="center",
		    va="bottom",
		    fontsize=12
		)

		ax.annotate(
		    "",
		    xy=(0.01, 1.),
		    xytext=(0.01, 0.08),
		    xycoords="axes fraction",
		    arrowprops=dict(
		        arrowstyle="-|>",
		        color="black",
		        lw=1.5,
		        mutation_scale=14
		    )
		)

		for frmt in frmts:
			fig.savefig(
				out_dir + "shap_beeswarm"+frmt,
				dpi=150,
				bbox_inches="tight"
			)
		print("beeswarm plot saved")
		plt.close(fig)
		gc.collect()



def plot_all(id=None):
	'''
	Use all the functions above and plot everything
	'''

	csv_files = list(Path(".").rglob("*.csv"))
	if id is not None:
		csv_files = [f for f in csv_files if id in str(f)]
	print(csv_files)
	for csv_file in csv_files:
		df = pd.read_csv(csv_file)
		for col in ['loss', 'accuracy']:
			plot_running_metric(df, col, str(csv_file.parent)+'/')

	npz_files = list(Path(".").rglob("*.npz"))
	if id is not None:
		npz_files = [f for f in npz_files if id in str(f)]

	npz_files_by_parent = {}
	for npz_file in npz_files:
		parent = npz_file.parent
		parentparent = parent.parent
		name = npz_file.name.lower()
		if "train" in name:
			split = "train"
		elif "val" in name:
			split = "val"
		elif "test" in name:
			split = "test"
		else:
			print(	f'Error: expected to find "train", "val", or "test" npzs '
					f'in the folder {parent}/')
			sys.exit(1)
		npz_files_by_parent.setdefault(str(parentparent) + "/", {}) \
						.setdefault(str(parent) + "/", {})[split] = np.load(npz_file)


	print(npz_files_by_parent)

	for parentparent, subdict in npz_files_by_parent.items():

		for parent, files in subdict.items():
			plot_roc_curve(y_pred=files['test']['y_pred'], y_true=files['test']['y_true'], outdir=parent)
			plot_scores(train_pred=files['train']['y_pred'], train_true=files['train']['y_true'],
						test_pred=files['test']['y_pred'], test_true=files['test']['y_true'], outdir=parent)
			_ = compute_purity(test_pred=files['test']['y_pred'], test_true=files['test']['y_true'], outdir=parent)
		if id!='002' and id!='003':
			plot_purity_2d_scan(subdict, parentparent)





if __name__ == "__main__":
	plot_all()

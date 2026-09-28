from matplotlib import pyplot as plt
import matplotlib as mpl
custom_colours = ["indianred", "cornflowerblue", "darkseagreen", "peru", "lightpink", "khaki", "mediumpurple", "grey"]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colours)
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
import os
import argparse
import hist
import mplhep as mh
from particle import Particle

def make_histograms(var, data_pre, data_post, hprops, log_flag=False):
	"""
	Utility: produce the two histograms pre trigger / post trigger
	"""
	
	pre = data_pre[var]
	post = data_post[var]

	if log_flag:
		pre = np.log(pre)
		post = np.log(post)

	if var == 'deta':
		pre = np.abs(pre)
		post = np.abs(post)

	# Remove extreme values
	pre = np.where(np.abs(pre) < 1e7, pre, np.nan)
	post = np.where(np.abs(post) < 1e7, post, np.nan)

	# Define binning
	if hprops['bw'] is not None:
		bw = hprops['bw']

		vis_range = np.arange(
			hprops['min'],
			hprops['max'] + bw,
			bw
		)

		valid_values = np.concatenate([
			pre[np.isfinite(pre)],
			post[np.isfinite(post)]
		])

		abs_min = np.min(valid_values)
		abs_max = np.max(valid_values)

		inv_min = vis_range[0] + bw * np.floor(
			(abs_min - vis_range[0]) / bw
		)
		inv_max = vis_range[0] + bw * np.ceil(
			(abs_max - vis_range[0]) / bw
		)

		bs = np.arange(inv_min, inv_max + bw, bw)
	else:
		bs = 'auto'

	# Get common bin edges
	_, edges = np.histogram(
		pre[np.isfinite(pre)],
		bins=bs
	)

	# Create histograms
	h1 = (
		hist.new.Variable(edges, name="post_data")
		.Weight()
		.fill(post)
		/ len(pre)
	)

	h2 = (
		hist.new.Variable(edges, name="pre_data")
		.Weight()
		.fill(pre)
		/ len(pre)
	)

	return h1, h2



frmts = ['.pdf', '.png']

m_combs = ['MX_1000_MY_125',
		'MX_500_MY_125', 'MX_700_MY_125', 'MX_1400_MY_125', 'MX_2000_MY_125',
		'MX_1000_MY_90', 'MX_1000_MY_200', 'MX_1000_MY_400', 'MX_1000_MY_600']

parser = argparse.ArgumentParser(description="Make purity plots.")
parser.add_argument("-b",  action="store_true", default=False, help="Purity 2D Plots")
parser.add_argument("-d",  action="store_true", default=False, help="Purity Distribution Plots")
parser.add_argument("-f",  action="store_true", default=False, help="Jet Flavor Plots")
parser.add_argument("-s",  action="store_true", default=False, help="DNN Score cut plots")
parser.add_argument("--mjj", action="store_true", help="Choose mjj-based selection of the VBF Jet pair.")
parser.add_argument("--dnn", action="store_true", help="Choose DNN-based selection of the VBF Jet pair.")
parser.add_argument("--comp", action="store_true", help="Compare mjj-based and DNN-based selection of the VBF Jet pair. Can only be used after having run the analysis on both.")
#parser.add_argument("--debug", action="store_true", help="Debug: runs script on just a few events.")
args = parser.parse_args()


if sum([args.mjj, args.dnn, args.comp]) != 1:
	parser.error("You must specify either --mjj, --dnn, or --comp.")
if args.dnn:
	root_dir = 'dnn_sel/'
elif args.mjj:
	root_dir = 'mjj_sel/'
elif args.comp:
	root_dir = 'mjj_dnn_comp/'


# --- Purity 2D Plots ---

if args.b:

	mh.style.use('CMS')
	
	print('--- Making Purity 2D Plots ---')

	plot_folder = root_dir+'plots/purity_plots/'
	os.makedirs(plot_folder, exist_ok=True)

	pur_counts = pd.read_csv(root_dir+'csvs/VBF.csv')

	refs =  []
	percs = []

	for m_comb in m_combs:
		print('Doin', m_comb, '...')

		DeltaRs = np.load(root_dir+f'npzs/npzs_VBF_{m_comb}/JetMatches_blwp.npz')

		DeltaR_1 = DeltaRs['DeltaR_1']
		DeltaR_2 = DeltaRs['DeltaR_2']

		print(sum(np.isnan(DeltaR_1)), sum(np.isnan(DeltaR_2)))
		mask_nan = ~np.isnan(DeltaR_1) & ~np.isnan(DeltaR_2)
		DeltaR_1 = DeltaR_1[mask_nan]
		DeltaR_2 = DeltaR_2[mask_nan]

		row = pur_counts[pur_counts['sample'] == m_comb]
		purity_test_count = row['purity_test'].iloc[0]
		pur_denom = row['purity_denominator'].iloc[0]
		perc = purity_test_count/pur_denom
		refs.append(m_comb)
		percs.append(perc)
		pur_count = f"Purity={perc:.3f}"

		subfold = 'scatt/'
		os.makedirs(plot_folder+subfold, exist_ok=True)
		fig, ax = plt.subplots(layout='constrained')
		ax.scatter(DeltaR_1, DeltaR_2, s=1)
		ax.set_aspect('equal')
		ax.set_xlabel(r'$\Delta R_1$')
		ax.set_ylabel(r'$\Delta R_2$')
		ax.text(0.7, 0.95, pur_count, transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=3))
		ax.xaxis.set_minor_locator(MultipleLocator(0.4))
		ax.yaxis.set_minor_locator(MultipleLocator(0.4))
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.yaxis.set_major_locator(MultipleLocator(2))
		m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV, $M_Y={m_comb.split("_")[3]}$ GeV'
		ax.set_title(r'Scatter plot of $quark - VBF\, jet$ distances'+'\n'+m_comb_title)
		for frmt in frmts:
			fig.savefig(plot_folder + subfold + 'purity_scatt_' + m_comb + frmt, bbox_inches='tight')
		plt.close()

		subfold = 'hist/'
		os.makedirs(plot_folder+subfold, exist_ok=True)
		fig, ax = plt.subplots(layout='constrained')
		bins = np.arange(0., np.max([DeltaR_1, DeltaR_2])+0.2, 0.2)
		H, xedges, yedges = np.histogram2d(DeltaR_1, DeltaR_2, bins=(bins, bins))
		norm = LogNorm(vmin=H[H > 0].min(), vmax=H.max())
		h = ax.hist2d(DeltaR_1, DeltaR_2, bins=(bins, bins), cmap='gist_heat_r', norm=norm)
		ax.axhline(0.4, color='k')
		ax.axvline(0.4, color='k')
		ax.set_aspect('equal')
		ax.set_xlabel(r'$\Delta R_1$')
		ax.set_ylabel(r'$\Delta R_2$')
		ax.text(0.7, 0.95, pur_count, transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=3))
		ax.xaxis.set_minor_locator(MultipleLocator(0.4))
		ax.yaxis.set_minor_locator(MultipleLocator(0.4))
		ax.xaxis.set_major_locator(MultipleLocator(2))
		ax.yaxis.set_major_locator(MultipleLocator(2))
		m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV, $M_Y={m_comb.split("_")[3]}$ GeV'
		ax.set_title(r'2D distribution of quark-VBFjet distances'+'\n'+m_comb_title)
		fig.colorbar(h[3], ax=ax, label='Counts')
		for frmt in frmts:
			fig.savefig(plot_folder + subfold + 'purity_hist_' + m_comb + frmt, bbox_inches='tight')
		plt.close()
		print(H[0, 0] / H.sum())

		subfold = 'hist_zoom/'
		os.makedirs(plot_folder+subfold, exist_ok=True)
		fig, ax = plt.subplots(layout='constrained')
		mask = (DeltaR_1 < 2.) & (DeltaR_2 < 2.)
		DeltaR_1_zoom = DeltaR_1[mask]
		DeltaR_2_zoom = DeltaR_2[mask]
		bins = np.arange(0., np.max([DeltaR_1_zoom, DeltaR_2_zoom])+0.1, 0.1)
		h = ax.hist2d(DeltaR_1_zoom, DeltaR_2_zoom, bins=(bins, bins), cmap='gist_heat_r', norm=norm)
		ax.axhline(0.4, color='k')
		ax.axvline(0.4, color='k')
		ax.set_aspect('equal')
		ax.set_xlabel(r'$\Delta R_1$')
		ax.set_ylabel(r'$\Delta R_2$')
		ax.text(0.7, 0.95, pur_count, transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=3))
		ax.xaxis.set_minor_locator(MultipleLocator(0.1))
		ax.yaxis.set_minor_locator(MultipleLocator(0.1))
		ax.xaxis.set_major_locator(MultipleLocator(0.4))
		ax.yaxis.set_major_locator(MultipleLocator(0.4))
		m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV, $M_Y={m_comb.split("_")[3]}$ GeV'
		ax.set_title(r'2D histogram of quark-VBFjet distances'+'\n'+m_comb_title+'\n'+'Magnified')
		fig.colorbar(h[3], ax=ax, label='Counts')
		for frmt in frmts:
			fig.savefig(plot_folder + subfold + 'purity_hist_zoom_' + m_comb + frmt, bbox_inches='tight')
		plt.close()

	print(refs)
	print(percs)
	print()



# --- Purity Distribution Plots ---

if args.d:

	print('--- Making Purity Distribution Plots ---')

	mh.style.use('CMS')

	label_dict = {'mjj': r'$m_{jj}$', 'ht': r'$H_t$', 'deta': r'$\left|\Delta\eta_{jj}\right|$'}

	var_and_hprops = {	'mjj': 		{'min': 100, 			'max': 5100, 			'bw': 100},
						'deta': 	{'min': 0,	 			'max': 9,	 			'bw': 0.15}}


	for m_comb in m_combs:
		in_dir = root_dir+f'npzs/npzs_VBF_{m_comb}/'
		npz_names = os.listdir(in_dir)

		plots_dir = root_dir+f'plots/purity_distr_plots/purity_distr_plots_{m_comb}/'
		os.makedirs(plots_dir, exist_ok=True)

		data = {}
		for npz_name in npz_names:
			with np.load(in_dir+npz_name) as file:
				data[npz_name[:-4]] = {k: file[k] for k in list(file.keys())}

		ek = 'pre_trg'
		data_pre = list({ek: data.pop('blwp_evts')}.values())[0]
		data_post = list({ek: data.pop('blwp_pure_evts')}.values())[0]

		for var, hprops in var_and_hprops.items():
			print(var)
			log_flag = True if var[:3]=='log' else False

			varname = var
			if log_flag:
				var = var[4:]

			h1, h2 = make_histograms(var, data_pre, data_post, hprops, log_flag=log_flag)

			fig, ax_main, ax_comp = mh.comp.hists(
				h1,
				h2,
				comparison='ratio',
				xlabel=(var if var not in label_dict.keys() else label_dict[var])+('' if (log_flag or var=='deta') else ' [GeV]'),
				ylabel='Efficiency',
				h1_label='Correctly matched VBF events',
				h2_label='VBF events',
				markersize=15
			)
			ax_comp.set_xlim(hprops['min'], hprops['max'])
			ax_main.set_xlim(hprops['min'], hprops['max'])
			ax_comp.set_ylim(0, 1)
			ax_comp.set_ylabel(r'$\frac{\mathrm{Correctly\ matched}}{\mathrm{Total}}$')
			mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax_main)

			leg = ax_main.get_legend()
			m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV, $M_Y={m_comb.split("_")[3]}$ GeV'
			leg.set_title(m_comb_title, prop={'weight': 'bold'})

			out_dir = plots_dir
			os.makedirs(out_dir, exist_ok=True)
			figname = out_dir+varname
			for frmt in frmts:
				fig.savefig(figname+frmt, bbox_inches='tight')
			print(figname+' saved')
			plt.close()

	print()


# --- Purity Distribution Comparison Plots ---

if args.comp:

	print('--- Making Purity Distribution Comparison Plots ---')

	mh.style.use('CMS')

	label_dict = {'mjj': r'$m_{jj}$', 'ht': r'$H_t$', 'deta': r'$\left|\Delta\eta_{jj}\right|$'}

	var_and_hprops = {	'mjj': 		{'min': 100, 			'max': 5100, 			'bw': 100},
						'deta': 	{'min': 0,	 			'max': 9,	 			'bw': 0.15}}

	for m_comb in m_combs:
		in_dir_mjj = f'mjj_sel/npzs/npzs_VBF_{m_comb}/'
		npz_names_mjj = os.listdir(in_dir_mjj)
		in_dir_dnn = f'dnn_sel/npzs/npzs_VBF_{m_comb}/'
		npz_names_dnn = os.listdir(in_dir_dnn)

		plots_dir = root_dir+f'plots/purity_distr_plots/purity_distr_plots_{m_comb}/'
		os.makedirs(plots_dir, exist_ok=True)

		data_mjj = {}
		for npz_name in npz_names_mjj:
			with np.load(in_dir_mjj+npz_name) as file:
				data_mjj[npz_name[:-4]] = {k: file[k] for k in list(file.keys())}
		data_dnn = {}
		for npz_name in npz_names_dnn:
			with np.load(in_dir_dnn+npz_name) as file:
				data_dnn[npz_name[:-4]] = {k: file[k] for k in list(file.keys())}

		ek = 'pre_trg'
		data_mjj_pre = list({ek: data_mjj.pop('blwp_evts')}.values())[0]
		data_mjj_post = list({ek: data_mjj.pop('blwp_pure_evts')}.values())[0]
		data_dnn_pre = list({ek: data_dnn.pop('blwp_evts')}.values())[0]
		data_dnn_post = list({ek: data_dnn.pop('blwp_pure_evts')}.values())[0]

		for var, hprops in var_and_hprops.items():
			print(var)
			log_flag = True if var[:3]=='log' else False

			varname = var
			if log_flag:
				var = var[4:]

			h1_mjj, h2_mjj = make_histograms(var, data_mjj_pre, data_mjj_post, hprops, log_flag=log_flag)
			h1_dnn, h2_dnn = make_histograms(var, data_dnn_pre, data_dnn_post, hprops, log_flag=log_flag)

			fig, ax = plt.subplots()

			colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
			mh.comp.comparison(
				h1_mjj,
				h2_mjj,
				comparison='ratio',
				xlabel=(var if var not in label_dict.keys() else label_dict[var])+('' if (log_flag or var=='deta') else ' [GeV]'),
				#ylabel='Efficiency',
				h1_label='Correctly matched VBF events',
				h2_label='VBF events',
				markersize=15,
				ax = ax,
				color=colors[2],
				label=r'$m_{jj}$'
			)
			mh.comp.comparison(
				h1_dnn,
				h2_dnn,
				comparison='ratio',
				xlabel=(var if var not in label_dict.keys() else label_dict[var])+('' if (log_flag or var=='deta') else ' [GeV]'),
				#ylabel='Efficiency',
				h1_label='Correctly matched VBF events',
				h2_label='VBF events',
				markersize=15,
				ax = ax,
				color=colors[5],
				label='DNN'
			)
			ax.set_xlim(hprops['min'], hprops['max'])
			ax.set_ylim(0, 1)
			ax.set_ylabel(r'$\frac{\mathrm{Correctly\ matched}}{\mathrm{Total}}$')
			mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)

			leg = ax.legend(fontsize=32)
			m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV, $M_Y={m_comb.split("_")[3]}$ GeV'
			leg.set_title(m_comb_title, prop={'weight': 'bold'})

			out_dir = plots_dir
			os.makedirs(out_dir, exist_ok=True)
			figname = out_dir+varname
			for frmt in frmts:
				fig.savefig(figname+frmt, bbox_inches='tight')
			print(figname+' saved')
			plt.close()

	print()



# --- Jet Flavor Plots ---

if args.f:

	print('--- Making Jet Flavor Plots ---')

	# import matplotlib as mpl
	# mpl.rcdefaults()
	mh.style.use('CMS')

	def pdgId_to_name(x):
		l = []
		for id in x:
			if id == 0:
				l.append('x')
			else:
				l.append(Particle.from_pdgid(id).name)
		return l

	for m_comb in m_combs:
		in_dir = root_dir+f'npzs/npzs_VBF_{m_comb}/'
		npz_names = os.listdir(in_dir)

		plots_dir = root_dir+f'plots/jet_flavour_plots/jet_flavour_plots_{m_comb}/'
		os.makedirs(plots_dir, exist_ok=True)

		jet_matches = np.load(in_dir+'JetMatches_blwp.npz')
		jetFlav_1 = jet_matches['Jet1_flav']
		jetFlav_2 = jet_matches['Jet2_flav']


		for i, flavs in enumerate([jetFlav_1, jetFlav_2]):
			fig, ax = plt.subplots()
			flavs = pdgId_to_name(np.abs(flavs))
			values, counts = np.unique(flavs, return_counts=True)
			counts = counts / counts.sum()

			base = np.array([v for v in values if '~' not in v])

			# matter = np.array([counts[values == b][0] if np.any(values == b) else 0	for b in base])
			# antimatter = np.array([counts[values == b + '~'][0] if np.any(values == b + '~') else 0	for b in base])

			x = np.arange(len(base))

			# bars_matter = ax.bar(x, matter, width=0.75, label='Matter')
			# bars_antimatter = ax.bar(x, antimatter, width=0.75, label='Antimatter', bottom=matter)
			bars = ax.bar(x, counts, width=0.75)

			# ax.bar_label(bars_antimatter, fmt='%.3f', padding=3, fontsize=16)
			ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=16)

			ax.set_xticks(x)
			ax.set_xticklabels(base)

			#ax.legend() # bbox_to_anchor=(0.5, 0.9))

			ax.set_xlabel('Flavour')
			ax.set_ylabel('Fraction')

			m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV, $M_Y={m_comb.split("_")[3]}$ GeV'
			# r'$m_{jj}$ chosen jets flavour (Jet_partonFlavour)'
			ax.set_title(m_comb_title+'\n'+('First' if i == 0 else 'Second' if i == 1 else '///')+' Jet')

			for frmt in frmts:
				figname = plots_dir+f'jet_flavour_{m_comb}'+f'_{(i%2)+1}'+frmt
				fig.savefig(figname, bbox_inches='tight')
				print(figname+' saved')

	print()



# --- DNN Score cut plots ---

if args.dnn and args.s:

	mh.style.use('CMS')

	print('Making DNN Score cut plots')

	for m_comb in m_combs:
		in_dir = root_dir+f'vbf_purity_parquets/'
		in_dir_ggF = root_dir+f'dnn_eff/'
		df = pd.read_parquet(in_dir+f'purity_{m_comb}.parquet')
		df_ggF = pd.read_parquet(in_dir_ggF+f'dnn_eff_ggF_{m_comb}.parquet')

		plots_dir = root_dir+f'plots/purity_dnn_cut_plots/'
		os.makedirs(plots_dir, exist_ok=True)

		thresholds = np.linspace(0, 1, 1000+1)

		scores = df['selpair_dnnscore'].to_numpy()
		is_vbf = df['is_vbf'].to_numpy()
		n_evts = np.array([(scores > t).mean() for t in thresholds])
		fractions = np.array([
			is_vbf[scores > t].mean() if np.any(scores > t) else np.nan
			for t in thresholds
		])

		df_ggF = df_ggF[df_ggF['passed']]
		scores_ggF = df_ggF['score']
		n_evts_ggF = np.array([(scores_ggF > t).mean() for t in thresholds])

		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

		fig, ax = plt.subplots()
		ax.plot(thresholds, fractions, lw=3, label='Purity')
		ax.plot(thresholds, n_evts, lw=3, label='Fraction of VBF events passing the cut')
		ax.plot(thresholds, n_evts_ggF, lw=3, label='Fraction of ggF events passing the cut', color='forestgreen')
		ax.set_xlabel('DNN-score cut')
		ax.set_ylabel(' ')
		ax.grid()
		ax.set_aspect('equal')
		ax.legend()
		m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV, $M_Y={m_comb.split("_")[3]}$ GeV'
		ax.text(0., 0.175, m_comb_title, ha='left', va='bottom')
		mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
		for frmt in frmts:
			figname = plots_dir+f'purity_vs_dnn_score_cut_{m_comb}'+frmt
			fig.savefig(figname, bbox_inches='tight')
			print(figname, 'saved')
		plt.close(fig)

	for m_comb in m_combs:
		in_dir = root_dir+f'vbf_purity_parquets/'
		in_dir_ggF = root_dir+f'dnn_eff/'
		in_dir_npz = root_dir+f'npzs/npzs_VBF_{m_comb}/'

		df = pd.read_parquet(in_dir+f'purity_{m_comb}.parquet')
		df_ggF = pd.read_parquet(in_dir_ggF+f'dnn_eff_ggF_{m_comb}.parquet')
		npz = np.load(in_dir_npz+'JetMatches_blwp.npz')

		plots_dir = root_dir+f'plots/purity_dnn_cut_plots/'
		os.makedirs(plots_dir, exist_ok=True)

		thresholds = np.linspace(0, 1, 1000+1)

		scores = df['selpair_dnnscore'].to_numpy()
		is_vbf = df['is_vbf'].to_numpy()
		n_evts = np.array([(scores > t).mean() for t in thresholds])
		fractions = np.array([
			is_vbf[scores > t].mean() if np.any(scores > t) else np.nan
			for t in thresholds
		])

		df_ggF = df_ggF[df_ggF['passed']]
		scores_ggF = df_ggF['score']
		n_evts_ggF = np.array([(scores_ggF > t).mean() for t in thresholds])

		flav1, flav2 = npz['Jet1_flav'], npz['Jet2_flav']
		df['flav1'] = flav1
		df['flav2'] = flav2
		flav1_fracs = [
			((df["flav1"] == 21) & (df["selpair_dnnscore"] >= threshold)).mean()
			for threshold in thresholds
		]
		flav2_fracs = [
			((df["flav2"] == 21) & (df["selpair_dnnscore"] >= threshold)).mean()
			for threshold in thresholds
		]

		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

		fig, ax = plt.subplots()
		ax.plot(thresholds, fractions, lw=3, label='Purity')
		ax.plot(thresholds, n_evts, lw=3, label='VBF events')
		ax.plot(thresholds, n_evts_ggF, lw=3, label='ggF events', color='forestgreen')#colors[4])
		ax.plot(thresholds, flav1_fracs, lw=3, label='Gluonic jet 1', color='dimgrey')
		ax.plot(thresholds, flav2_fracs, lw=3, label='Gluonic jet 2', color='saddlebrown')
		ax.set_xlabel('DNN-score cut')
		ax.grid()
		ax.set_aspect('equal')
		m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV,'+'\n'+rf'$M_Y={m_comb.split("_")[3]}$ GeV'
		ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title=m_comb_title)
		mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
		for frmt in frmts:
			figname = plots_dir+f'purity_vs_dnn_score_cut_{m_comb}_v2'+frmt
			fig.savefig(figname, bbox_inches='tight')
			print(figname, 'saved')
		plt.close(fig)

	for m_comb in m_combs:
		in_dir = root_dir+f'vbf_purity_parquets/'
		in_dir_npz = root_dir+f'npzs/npzs_VBF_{m_comb}/'

		df = pd.read_parquet(in_dir+f'purity_{m_comb}.parquet')
		npz = np.load(in_dir_npz+'JetMatches_blwp.npz')

		plots_dir = root_dir+f'plots/purity_dnn_cut_plots/'
		os.makedirs(plots_dir, exist_ok=True)

		thresholds = np.linspace(0, 1, 1000+1)

		flav1, flav2 = npz['Jet1_flav'], npz['Jet2_flav']
		df['flav1'] = flav1
		df['flav2'] = flav2
		flav1_fracs = [
			((df["flav1"] == 21) & (df["selpair_dnnscore"] >= threshold)).mean()
			for threshold in thresholds
		]
		flav2_fracs = [
			((df["flav2"] == 21) & (df["selpair_dnnscore"] >= threshold)).mean()
			for threshold in thresholds
		]

		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

		fig, ax = plt.subplots()
		ax.plot(thresholds, flav1_fracs, lw=3, label='Jet 1', color='dimgrey')
		ax.plot(thresholds, flav2_fracs, lw=3, label='Jet 2', color='saddlebrown')
		ax.set_ylim(-0.01, 0.23)
		ax.set_xlabel('DNN-score cut')
		ax.set_ylabel('Gluon fraction')
		ax.grid()
		ax.set_box_aspect(1)
		m_comb_title = rf'$M_X={m_comb.split("_")[1]}$ GeV, $M_Y={m_comb.split("_")[3]}$ GeV'
		ax.legend(title=m_comb_title, alignment="right")
		mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
		for frmt in frmts:
			figname = plots_dir+f'purity_vs_dnn_score_cut_{m_comb}_v3'+frmt
			fig.savefig(figname, bbox_inches='tight')
			print(figname, 'saved')
		plt.close(fig)

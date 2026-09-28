import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
custom_colours = ["indianred", "cornflowerblue", "darkseagreen", "peru", "lightpink", "khaki", "mediumpurple", "grey"] # gold-khaki
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colours)
custom_markers = ['s', 'o', 'P', 'X', 'D', '^', 'v']
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import hist
import mplhep as mh
mh.style.use('CMS')

parser = argparse.ArgumentParser(description="Make plots of differential efficiency.")
parser.add_argument("--sample", type=str, help="Sample to read (for the moment: QCD (bkg)")
parser.add_argument("--mjj", action="store_true", help="Choose mjj-based selection of the VBF Jet pair.")
parser.add_argument("--dnn", action="store_true", help="Choose DNN-based selection of the VBF Jet pair.")
args = parser.parse_args()

if sum([args.mjj, args.dnn]) != 1:
	parser.error("You must specify either --mjj or --dnn.")
if args.dnn:
	root_dir = 'dnn_sel/'
elif args.mjj:
	root_dir = 'mjj_sel/'


triggers = {    0: {'trigger_name': "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5", 'label': "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5"},       # Analogous to Run 2    # 000
				1: {'trigger_name': "HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55", 'label': "HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55"},                                       # Analogous to Run 3      # 001
				# Standard stream
				2: {'trigger_name': "HLT_VBF_DiPFJet125_45_Mjj1050", 'label': "HLT_VBF_DiPFJet125_45_Mjj1050"},                                                            # 002
				3: {'trigger_name': "HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60", 'label': "HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60"},                                               # 003
				# Parking stream
				4: {'trigger_name': "HLT_QuadPFJet103_88_75_15", 'label': "HLT_QuadPFJet103_88_75_15"},                                                                   # 004
				5: {'trigger_name': "HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1", 'label': "HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1"},                                         # 005
				6: {'trigger_name': "HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2", 'label': "HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2"}                                                 # 006
		}

trigger_names_ids = [d['trigger_name'] for d in triggers.values()]
'''
[
		# Example triggers
		"HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5", # Analogous to Run 2   # 000
		"HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55", # Targeting ggF HH                         # 001
		# VBF HH
		# Standard stream
		"HLT_VBF_DiPFJet125_45_Mjj1050",                                                        # 002
		"HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60",                                                # 003
		# Parking stream
		"HLT_QuadPFJet103_88_75_15",                                                            # 004
		"HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1",                                    # 005
		"HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2",                                          # 006
]
'''

trigger_labels = [d['label'] for d in triggers.values()]
'''
[
		"Run 2 Analogous",              # 000
		"ggF",                          # 001
		"VBF SS High En Cut",           # 002
		"VBF SS Double DiJet Cut",      # 003
		"PS No B Tag",              	# 004
		"VBF PS 2BTag VBF1",            # 005
		"VBF PS 2BTag VBF2",            # 006
]
'''
map_id_to_lab = dict(zip(trigger_names_ids, trigger_labels))

map_trg_to_marker = {t: m for t, m in zip(trigger_names_ids, custom_markers)}

label_dict = {'mjj': r'$m_{jj}$', 'ht': r'$H_t$', 'deta': r'$\left|\Delta\eta_{jj}\right|$'}





var_and_hprops = {	'ht': 		{'min': 100, 			'max': 2100, 			'bw': 50},
					#'log_ht': 	{'min': np.log(100), 	'max': np.log(2100), 	'bw': np.log(50)	},
					'mjj': 		{'min': 100, 			'max': 5100, 			'bw': 100},
					#'log_mjj': 	{'min': np.log(100),	'max': np.log(5100),	'bw': np.log(100)	},
					'deta': 	{'min': 0,	 			'max': 9,	 			'bw': 0.15}}


combs = [f[5:] for f in [entry.name for entry in os.scandir(root_dir+'npzs') if entry.is_dir()]]
if args.sample is not None:
	combs = [args.sample]


# Cycle for the combinations
for comb in combs:
	in_dir = root_dir+f'npzs/npzs_{comb}/'
	npz_names = os.listdir(in_dir)

	plots_dir = root_dir+f'plots/distr_plots/distr_plots_{comb}/'
	os.makedirs(plots_dir, exist_ok=True)

	data = {}
	for npz_name in npz_names:
		with np.load(in_dir+npz_name) as file:
			data[npz_name[:-4]] = {k: file[k] for k in list(file.keys())}

	ek = 'pre_trg'
	data_pre = list({ek: data.pop('pre_trg')}.values())[0]
	data_post = data

	data_post = {k: v for k, v in data_post.items()}

	trigger_names = list(data_post.keys())

	# Cycle for variables to plot
	for var, hprops in var_and_hprops.items():
		print(var)
		log_flag = True if var[:3]=='log' else False

		varname = var
		if log_flag:
			var = var[4:]

		hposts = []
		hposts_v2 = []
		trigger_names_ids_v2 = ['HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55', "HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1", "HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2"]

		# Trigger cycle
		for trg_name in trigger_names_ids:
			pre = data_pre[var]
			post = data_post[trg_name][var]

			if log_flag:
				pre = np.log(pre)
				post = np.log(post)

			if var == 'deta':
				pre = abs(pre)
				post = abs(post)

			pre = np.where(np.abs(pre) < 10e6, pre, np.nan) ##
			post = np.where(np.abs(post) < 10e6, post, np.nan)

			if hprops['bw'] is not None:
				vis_range = np.arange(hprops['min'], hprops['max'] + hprops['bw'], hprops['bw'])
				abs_min = np.min(np.concatenate([pre[np.isfinite(pre)], post[np.isfinite(post)]]))
				abs_max = np.max(np.concatenate([pre[np.isfinite(pre)], post[np.isfinite(post)]]))
				inv_min = vis_range[0] + hprops['bw'] * np.floor((abs_min - vis_range[0]) / hprops['bw'])
				inv_max = vis_range[0] + hprops['bw'] * np.ceil((abs_max - vis_range[0]) / hprops['bw'])
				bs = np.arange(inv_min, inv_max + hprops['bw'], hprops['bw'])
			else:
				bs = 'auto'
			counts, edges = np.histogram(pre[np.isfinite(pre)], bins=bs)

			h1 = hist.new.Variable(edges, name="post_data").Weight().fill(post) / len(pre)
			hposts.append(h1)
			if trg_name in trigger_names_ids_v2:
					hposts_v2.append(h1)
			h2 = hist.new.Variable(edges, name="pre_data").Weight().fill(pre) / len(pre)

			fig, ax_main, ax_comp = mh.comp.hists(
				h1,
				h2,
				comparison='ratio',
				xlabel=(var if var not in label_dict.keys() else label_dict[var])+('' if (log_flag or var=='deta') else ' [GeV]'),
				ylabel='Efficiency',
				h1_label='Post',
				h2_label='Pre',
				markersize=15
			)
			txt = mh.add_text(map_id_to_lab[trg_name], loc='upper left', ax=ax_main, pad=2.5, fontsize=16)
			if comb[:3] == 'QCD' and var == 'deta':
				ax_comp.set_xlim(hprops['min'], 6.)
				ax_main.set_xlim(hprops['min'], 6.)
			else:
				ax_comp.set_xlim(hprops['min'], hprops['max'])
				ax_main.set_xlim(hprops['min'], hprops['max'])
			ax_comp.set_ylim(0, 1)
			mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax_main)

			out_dir = plots_dir+varname+'/'
			os.makedirs(out_dir, exist_ok=True)
			figname = out_dir+varname+'_'+trg_name
			fig.savefig(figname+'.png', bbox_inches='tight')
			print(figname+' saved')
			plt.close()

		# Plot all triggers on the same figure
		fig, ax = plt.subplots()
		for hp, c, m in zip(hposts, custom_colours[:len(hposts)], custom_markers):
			mh.comp.comparison(hp, h2, h1_label=map_id_to_lab[trg_name], h2_label='Pre trg', comparison='ratio',
								xlabel=(var if var not in label_dict.keys() else label_dict[var])+('' if (log_flag or var=='deta') else ' [GeV]'), 
								ax=ax, color=c, elinewidth=0, linestyle='-', marker=m, markersize=7.5)
		scale = 0.75 / (h2.values()[np.argmax(h2.values())])
		mh.histplot(h2*scale, histtype='fill', alpha=0.3, color = 'grey', ax=ax)
		splits = comb.split('_')
		if splits[0] == 'VBF':
			pre_label = f'(MX={splits[2]} GeV, MY={splits[4]} GeV) VBF Signal '
		elif splits[0] == 'ggF':
			pre_label = f'(MX={splits[2]} GeV, MY={splits[4]} GeV) ggF Signal '
		elif splits[0] == 'QCD':
			pre_label = f'QCD Signal '
		hist_patch = Patch(facecolor='grey', edgecolor='grey', alpha=0.3,  label=pre_label + r'$\times$' + f'{int(scale)}')
		if comb[:3] == 'QCD' and var == 'deta':
			ax.set_xlim(hprops['min'], 6.)
		else:
			ax.set_xlim(hprops['min'], hprops['max'])
		ax.set_ylim(0, 1)
		ax.set_ylabel('Efficiency')
		mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
		handles = [
			Line2D([0], [0], color=c, marker=map_trg_to_marker[name], linestyle='-',
				label=map_id_to_lab[name])
			for c, name in zip(custom_colours[:len(hposts)], map_id_to_lab.keys())
		]
		handles += [hist_patch]
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.15,
						 box.width, box.height * 0.85])
		ax.legend(handles=handles, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15),
				  fancybox=True, shadow=True, )
		out_dir = plots_dir+varname+'/'
		os.makedirs(out_dir, exist_ok=True)
		figname = out_dir+varname+'_trgs'
		fig.savefig(figname+'.pdf', bbox_inches='tight')
		fig.savefig(figname+'.png', bbox_inches='tight')
		print(figname+' saved')

		# v2: presentation version with only Run 3 inclusive and the two VBF triggers

		fig, ax = plt.subplots()
		cust_cols = [custom_colours[1], custom_colours[5], custom_colours[6]]
		for hp, c, m in zip(hposts_v2, cust_cols, custom_markers):
			mh.comp.comparison(hp, h2, h1_label=map_id_to_lab[trg_name], h2_label='Pre trg', comparison='ratio',
								xlabel=(var if var not in label_dict.keys() else label_dict[var])+('' if (log_flag or var=='deta') else ' [GeV]'), 
								ax=ax, color=c, elinewidth=0, linestyle='-', marker=m, markersize=7.5)
		scale = 0.75 / (h2.values()[np.argmax(h2.values())])
		mh.histplot(h2*scale, histtype='fill', alpha=0.3, color = 'grey', ax=ax)
		splits = comb.split('_')
		if splits[0] == 'VBF':
			pre_label = f'(MX={splits[2]} GeV, MY={splits[4]} GeV) VBF Signal '
		elif splits[0] == 'ggF':
			pre_label = f'(MX={splits[2]} GeV, MY={splits[4]} GeV) ggF Signal '
		elif splits[0] == 'QCD':
			pre_label = f'QCD Signal '
		hist_patch = Patch(facecolor='grey', edgecolor='grey', alpha=0.3,  label=pre_label + r'$\times$' + f'{int(scale)}')
		if comb[:3] == 'QCD' and var == 'deta':
			ax.set_xlim(hprops['min'], 6.)
		else:
			ax.set_xlim(hprops['min'], hprops['max'])
		ax.set_ylim(0, 1)
		ax.set_ylabel('Efficiency')
		mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
		handles = [
			Line2D([0], [0], color=c, marker=map_trg_to_marker[name], linestyle='-',
				label=map_id_to_lab[name])
			for c, name in zip(cust_cols, map_id_to_lab.keys())
		]
		handles += [hist_patch]
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.15,
						 box.width, box.height * 0.85])
		ax.legend(handles=handles, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15),
				  fancybox=True, shadow=True, )#ncol=2)
		out_dir = plots_dir+varname+'/'
		os.makedirs(out_dir, exist_ok=True)
		figname = out_dir+varname+'_trgs'
		fig.savefig(figname+'_v2.pdf', bbox_inches='tight')
		fig.savefig(figname+'_v2.png', bbox_inches='tight')
		print(figname+' saved')


# Plot distributions after the VBF cut
if len(combs)>1:

	var_and_hprops['deta']['max'] = 9.
	var_and_hprops['deta']['min'] = (-1.) * var_and_hprops['deta']['max']

	plots_dir = root_dir+f'plots/distr_plots/distr_plots_after_vbf_cut/'
	os.makedirs(plots_dir, exist_ok=True)

	m = [s for s in combs if all(x in s for x in ['VBF', '1000', '125'])]
	if len(m) != 1:
		raise ValueError(f"Error: found too many VBF 1000 125 samples ({len(m)})")
	vbf_sample = np.load(root_dir+f'npzs/npzs_{m[0]}/vbfCut_evts.npz')
	vbf_sample_mjj = vbf_sample['mjj'][np.isfinite(vbf_sample['mjj'])]
	vbf_sample_deta = vbf_sample['deta'][np.isfinite(vbf_sample['deta'])]

	m = [s for s in combs if all(x in s for x in ['ggF', '1000', '125'])]
	if len(m) != 1:
		raise ValueError(f"Error: found too many ggF 1000 125 samples ({len(m)})")
	ggf_sample = np.load(root_dir+f'npzs/npzs_{m[0]}/vbfCut_evts.npz')
	ggf_sample_mjj = ggf_sample['mjj'][np.isfinite(ggf_sample['mjj'])]
	ggf_sample_deta = ggf_sample['deta'][np.isfinite(ggf_sample['deta'])]

	m = [s for s in combs if 'QCD' in s]
	if len(m) != 1:
		raise ValueError(f"Error: found too many QCD samples ({len(m)})")
	qcd_sample = np.load(root_dir+f'npzs/npzs_{m[0]}/vbfCut_evts.npz')
	qcd_sample_mjj = qcd_sample['mjj'][np.isfinite(qcd_sample['mjj'])]
	qcd_sample_deta = qcd_sample['deta'][np.isfinite(qcd_sample['deta'])]

	samples_vars = {
		'mjj': {'VBF': vbf_sample_mjj, 'ggF': ggf_sample_mjj, 'QCD': qcd_sample_mjj},
		'deta': {'VBF': vbf_sample_deta, 'ggF': ggf_sample_deta, 'QCD': qcd_sample_deta},
	}

	for var, xlab in zip(['mjj', 'deta'], [r'$m_{jj}$ [GeV]', r'$\Delta\eta_{jj}$']):

		hprops = var_and_hprops[var]
		if hprops['bw'] is not None:
			vis_range = np.arange(hprops['min'], hprops['max'] + hprops['bw'], hprops['bw'])
			abs_min = np.min(np.concatenate(list(samples_vars[var].values())))
			abs_max = np.max(np.concatenate(list(samples_vars[var].values())))
			inv_min = vis_range[0] + hprops['bw'] * np.floor((abs_min - vis_range[0]) / hprops['bw'])
			inv_max = vis_range[0] + hprops['bw'] * np.ceil((abs_max - vis_range[0]) / hprops['bw'])
			bs = np.arange(inv_min, inv_max + hprops['bw'], hprops['bw'])
		else:
			bs = 'auto'
		counts, edges = np.histogram(samples_vars[var]['VBF'], bins=bs)
		fig, ax = plt.subplots()
		h_vbf = hist.new.Variable(edges, name="VBF").Weight().fill(samples_vars[var]['VBF']) / len(samples_vars[var]['VBF'])
		h_ggf = hist.new.Variable(edges, name="ggF").Weight().fill(samples_vars[var]['ggF']) / len(samples_vars[var]['ggF'])
		h_qcd = hist.new.Variable(edges, name="QCD").Weight().fill(samples_vars[var]['QCD']) / len(samples_vars[var]['QCD'])
		mh.histplot(h_vbf, ax=ax, label='VBF')
		mh.histplot(h_qcd, ax=ax, label='ggF')
		mh.histplot(h_ggf, ax=ax, label='QCD')
		ax.legend()
		ax.set_xlabel(xlab)
		ax.set_xlim(hprops['min'], hprops['max'])
		mh.cms.label(llabel='Private work', rlabel='Simulation', ax=ax)
		fig.savefig(plots_dir+var+'.png', bbox_inches='tight')

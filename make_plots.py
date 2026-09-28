import numpy as np
import pandas as pd
import matplotlib as mpl
custom_colours = ["indianred", "cornflowerblue", "darkseagreen", "peru", "lightpink", "khaki", "mediumpurple", "grey"] # sienna-peru, gold-khaki
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colours)
custom_markers = ['s', 'o', 'P', 'X', 'D', '^', 'v']
from matplotlib import pyplot as plt
plt.rcParams.update(
	{
		"xtick.direction": "in", "ytick.direction": "in",
	"xtick.top": True, "ytick.right": True,
	"xtick.minor.ndivs": 'auto', "ytick.minor.ndivs": 'auto',
	"xtick.minor.visible": True, "ytick.minor.visible": True
	}
)
import matplotlib.patheffects as PathEffects
from operator import itemgetter
import argparse
import os
import re
from matplotlib.patches import Patch


'''
python make_plots.py -t ground; python make_plots.py -t VBF_park; python make_plots.py -t VBF_std; python make_plots.py -t VBF_std_VBF_park; python make_plots.py -t ground_VBF_std_VBF_park
'''
frmts = ['pdf', 'png']


triggers = {0: {'trigger_name': "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5", 'label': "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5"},	# Analogous to Run 2	# 000
			1: {'trigger_name': "HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55", 'label': "HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55"},					# Run3 Inclusive  	# 001
			# Standard stream
			2: {'trigger_name': "HLT_VBF_DiPFJet125_45_Mjj1050", 'label': "HLT_VBF_DiPFJet125_45_Mjj1050"},								# 002
			3: {'trigger_name': "HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60", 'label': "HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60"},						# 003
			# Parking stream
			4: {'trigger_name': "HLT_QuadPFJet103_88_75_15", 'label': "HLT_QuadPFJet103_88_75_15"},									# 004
			5: {'trigger_name': "HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1", 'label': "HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1"},						# 005
			6: {'trigger_name': "HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2", 'label': "HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2"}							# 006
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
	"Run 2 Analogous",		# 000
	"ggF",				# 001
	"VBF SS High En Cut",		# 002
	"VBF SS Double DiJet Cut",	# 003
	"PS No B Tag",				# 004
	"VBF PS 2BTag VBF1",		# 005
	"VBF PS 2BTag VBF2",		# 006
]
'''
map_id_to_lab = dict(zip(trigger_names_ids, trigger_labels))

map_trg_to_color = {t: c for t, c in zip(trigger_names_ids+['all'], custom_colours)}

map_trg_to_marker = {t: m for t, m in zip(trigger_names_ids, custom_markers)}


def preprocess(id, ratio_id=None):
	'''
	Preprocessing of data, returning the dataframes for mx and my scans
	ratio_id should contain only one trigger
	'''
	MX_ref = 1000
	MY_ref = 125

	df = pd.read_csv(f"csvs/mxmy_scan_{id}.csv", dtype="float")
	df = df.sort_values(by="MX").sort_values(by="MY")

	column_names = df.columns
	trigger_names = column_names[3:-2]
	trigger_names = trigger_names[:len(trigger_names)//2]

	for col in column_names[3:]:
		df[f"eff_{col}"] = df[col]/df['tot_n_evts']
	if ratio_id is not None:
		df_r = pd.read_csv(f"csvs/mxmy_scan_{ratio_id}.csv", dtype="float")
		df_r = df_r.sort_values(by="MX").sort_values(by="MY")
		for col in df_r.columns[3:]:
			df_r[f"eff_{col}"] = df_r[col]/df_r['tot_n_evts']
		for col in column_names[3:]:
			df[f"eff_{col}"] = df[f"eff_{col}"]/df_r[df_r.columns[7]]
			print(df[f"eff_{col}"])

	print(df.columns)
	print(df.to_string(header=False))

	scan_x = df[df.duplicated(subset=["MY"], keep=False)]
	scan_y = df[df.duplicated(subset=["MX"], keep=False)]

	return scan_x, scan_y, df, trigger_names


def preprocess_sens():
	'''
	Preprocessing for the sensitivity plots
	'''

	df_bkg = pd.read_csv(f"csvs/bkg_scan_all.csv")

	column_names = df_bkg.columns
	trigger_names = column_names[2:-2]
	trigger_names = trigger_names[:len(trigger_names)//2]

	for col in column_names[2:]:
		df_bkg[f"eff_{col}"] = df_bkg[col]/df_bkg['tot_n_evts']
	

	df_sig = pd.read_csv(f"csvs/mxmy_scan_ground_VBF_std_VBF_park.csv")

	column_names = df_sig.columns
	len_cols = len(column_names)
	trigger_names = column_names[3:-2]
	trigger_names = trigger_names[:len(trigger_names)//2]


	for col in column_names[3:]:
		df_sig[f"eff_{col}"] = df_sig[col]/df_sig['tot_n_evts']
	
	for col in df_sig.columns[len_cols:len_cols+len(trigger_names)]:
		df_sig[f"sens_{col[4:]}"] = df_sig[col] / np.sqrt(df_bkg[col].iloc[0])

	print(df_sig.columns)
	print(df_sig.to_string(header=False))

	scan_x = df_sig[df_sig.duplicated(subset=["MY"], keep=False)]
	scan_y = df_sig[df_sig.duplicated(subset=["MX"], keep=False)]

	return scan_x, scan_y, df_sig, df_bkg, trigger_names


def plot_1d_scan(grp, xlab, trg_set, title, img_folder, id, fig=None, custom_lab=None, ratio=False, colour_skip=0, sens=False, fancy=False, png=False):
	'''
	Plot the scan along either MX or MY
	'''
	if fig is None:
		fig, ax = plt.subplots()
	else:
		ax = fig.axes[0]
	ax.set_box_aspect()
	ax.set_axisbelow(True)
	ax.grid()
	ax.set_prop_cycle(color=custom_colours[colour_skip:])
	for ylab in trg_set:
		if custom_lab is None:
			if ylab == 'eff_any':
				lab = 'OR'
			elif ylab == 'eff_all':
				lab = 'AND'
			else:
				lab = ylab
				if lab[:4] == 'eff_':
					lab = lab[4:]
				if lab[:5] == 'excl_':
					lab = lab[5:]
				if lab[:5] == 'sens_':
					lab = lab[5:]
				lab = map_id_to_lab[lab]
		else:
			lab = custom_lab
		clr = None
		if lab == 'Targeting ggF':
			clr = custom_colours[1]
		elif lab == "VBF Standard Stream":
			clr = custom_colours[3]
		elif lab == "VBF Parking Stream":
			clr = custom_colours[5] 
		elif lab == f"{trigger_labels[1]} OR {trigger_labels[2]}":
			clr = custom_colours[2] 
		elif lab == f"{trigger_labels[1]} OR {trigger_labels[4]}":
			clr = custom_colours[4] 
		mlab = lab[5:] if lab.startswith("OR   ") else lab
		scatt = ax.scatter(grp[xlab], grp[ylab], zorder=1, label=lab, color=clr, marker=('o' if mlab not in map_trg_to_marker.keys() else map_trg_to_marker[mlab]))
		if fancy:
			_ = ax.plot(grp[xlab], grp[ylab], zorder=1, color=clr)
		if not fancy:
			for i, p in grp.iterrows():
				if round(p[ylab], 3)==1.:
					continue
				text = ax.annotate(
					f"{p[ylab]:.3f}",
					(p[xlab], p[ylab]),
					textcoords="offset points",
					xytext=(5, 5),
					fontsize=8,
					color=scatt.get_facecolor(),
				)
				text.set_path_effects([PathEffects.withStroke(linewidth=0.5, foreground='black')]) ###


	if True:# len(trg_set) > 1:
		ax.legend(
			fontsize=8,
			loc='upper center',
			bbox_to_anchor=(0.5, -0.15),
			fancybox=True,
			shadow=True
		)
		fig.subplots_adjust(bottom=0.22)


	if xlab == 'MX':
		xlabel = r'$M_X$ (GeV)'
	elif xlab == 'MY':
		xlabel = r'$M_Y$ (GeV)'
	ax.set_xlabel(xlabel)
	ax.set_ylabel('Ratio wrt. Run 2' if ratio else r'Sensitivity $\epsilon_S/\sqrt{\epsilon_B}$' if sens else 'Efficiency')
	ax.set_ylim(0., 7. if ratio else 10. if sens else 1.)
	ax.set_title(title)
	for frmt in frmts+['png']:
		fig.savefig(img_folder+id+("_ratio" if ratio else "")+"_1Dscan_"+xlab+"_"+title.split()[0]+f".{frmt}", bbox_inches='tight', dpi=500)
	return fig


def plot_2d_scan(df, values, title, img_folder, id, fancy=False):
	'''
	Plot the scan in the MX vs MY plane
	'''
	fig, ax = plt.subplots()
	ax.set_axisbelow(True)
	ax.grid()
	scatter = ax.scatter(df['MX'], df['MY'], c=df[values], cmap='turbo', vmin=0, vmax=1)
	cbar = fig.colorbar(scatter)
	cbar.set_ticks(np.arange(0., 1.+0.1, 0.1))
	if not fancy:
		for i, p in df.iterrows():
			ax.annotate(
				f"{p[values]:.3f}",
				(p["MX"], p["MY"]),
				textcoords="offset points",
				xytext=(5, 5),
				fontsize=8,
			)
	ax.set_xlabel(r"$M_X$ (GeV)")
	ax.set_ylabel(r"$M_Y$ (GeV)")
	ax.set_title(title)
	for frmt in frmts:
		fig.savefig(img_folder+id+"_2Dscan_"+title.split(' ')[0]+f".{frmt}", bbox_inches='tight', dpi=500)


def plot_2d_3d_scan(df, title, img_folder, id, fancy=False):
	'''
	Plot the scan in the MX vs MY plane visualized as contributions in 3d bar
	Supposed to work only for pairs of triggers
	'''
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	labels = ['eff_all', *['eff_excl_'+t for t in trigger_names]]
	dx = 0.25*(max(df["MX"])-min(df["MX"]))/6.
	dy = 0.25*(max(df["MY"])-min(df["MY"]))/6.
	zpos = np.zeros_like(df[labels[0]])
	handles = []
	labs = []
	for label in labels:
		if label == 'eff_any':
			lab = 'OR'
		elif label == 'eff_all':
			if len(labels)==3:
				lab = 'Both'
			else:
				lab = 'AND'
		else:
			lab = label
			if lab[:4] == 'eff_':
				lab = lab[4:]
			if lab[:5] == 'excl_':
				lab = lab[5:]
			lab = map_id_to_lab[lab]
		clr = map_trg_to_color[label.removeprefix("eff_excl_") if label.startswith("eff_excl_") else "all"]
		ax.bar3d(
			df["MX"], df["MY"], zpos,
			dx*np.ones_like(df["MX"]), dy*np.ones_like(df["MY"]), df[label],
			zsort='average', color=clr, #label=lab
		)
		handles.append(
			Patch(facecolor=clr, edgecolor=clr, label=lab)
		)
		labs.append(lab)
		print(label)
		zpos += df[label]
	if not fancy:
		for x, y, total in zip(df["MX"], df["MY"], zpos):
			ax.text(
				x+dx/2, y+dy/2, total+0.02, f"{total:.3f}", 'x',
				ha='center', va='bottom', zorder=10,
			)
	handles = handles[1:] + handles[:1]
	labels = labels[1:] + labels[:1]
	ax.legend(handles=handles, fontsize=6)
	ax.set_zlim(0., 1.)
	ax.set_xlabel(r"$M_X$ (GeV)")
	ax.set_ylabel(r"$M_Y$ (GeV)")
	ax.set_zlabel("Efficiency")
	for frmt in frmts+['png']:
		fig.savefig(img_folder+id+f"_2Dscan_3D.{frmt}", dpi=500)

def plot_2_contribs(df, title, img_folder, id, scan, fancy=False):
	'''
	Plot either MX or MY scan, showing the individual contributions
	Supposed to work only for pairs of triggers
	'''
	fig, ax = plt.subplots()
	labels = ['eff_all', *['eff_excl_'+t for t in trigger_names]]
	df_plot = df[np.isclose(df["MY" if scan=='x' else 'MX'], 125 if scan=='x' else 1000)].sort_values("MX" if scan=='x' else 'MY')
	bottom = np.zeros(len(df_plot))
	for label in labels:
			if label == 'eff_any':
					lab = 'OR'
			elif label == 'eff_all':
				if len(labels)==3:
					lab = 'Both'
				else:
					lab = 'AND'
			else:
				lab = label
				if lab[:4] == 'eff_':
						lab = lab[4:]
				if lab[:5] == 'excl_':
						lab = lab[5:]
				lab = map_id_to_lab[lab]
			clr = map_trg_to_color[label.removeprefix("eff_excl_") if label.startswith("eff_excl_") else "all"]
			ax.bar(
				np.arange(len(df_plot)),
				df_plot[label], width=0.4,
				bottom=bottom, color=clr,
				label=lab
			)
			bottom += df_plot[label].to_numpy()
	if not fancy:
		x_pos = np.arange(len(df_plot))
		for x, total in zip(x_pos, bottom):
			ax.text(
				x, total + 0.02,
				f"{total:.3f}",
				ha='center',
				va='bottom',
				zorder=10,
			)
	handles, labels = ax.get_legend_handles_labels()
	handles = handles[1:] + handles[:1]
	labels = labels[1:] + labels[:1]
	ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),)
	ax.set_xticklabels([int(x) for x in df_plot['MX' if scan=='x' else 'MY']])
	ax.set_ylim(0., 1.)
	ax.set_xlabel(r"$M_X$ (GeV)" if scan=='x' else r"$M_Y$ (GeV)")
	ax.set_ylabel("Efficiency")
	ax.set_title(title)
	for frmt in frmts+['png']:
			fig.savefig(img_folder+id+f"_2scan_{scan}.{frmt}", dpi=500, bbox_inches='tight')


def plot_1d_scan_comparison(img_folder, ids, labels, leg_title=None, fname_prefix=None, ratio_id=None, colour_skip=0, y_cut=None, fancy=False, skip_ids=None):
	'''
	Plot either MX or MY scan of efficiencies, comparing triggers
	Uses plot_1d_scan
	'''
	
	print(img_folder, ids, labels, (leg_title if leg_title is not None else ''), (fname_prefix if fname_prefix is not None else ''), '\n')
	fname_prefix = img_folder.split("/")[-2]
	fig_x, fig_y = None, None
	print(labels)
	for i, (id, lab) in enumerate(zip(ids, labels)):
		scan_x, scan_y, df, trigger_names = preprocess(id, ratio_id)
		if skip_ids is None or i not in skip_ids:
			fig_x = plot_1d_scan(scan_x, "MX", ["eff_any"], r"OR Efficiency - $M_X$ scan (fixed $M_Y=125$ GeV)", img_folder, fname_prefix, fig=fig_x, custom_lab=lab, ratio=(True if ratio_id is not None else False), colour_skip=colour_skip, fancy=fancy)
			fig_y = plot_1d_scan(scan_y, "MY", ["eff_any"], r"OR Efficiency - $M_Y$ scan (fixed $M_X=1000$ GeV)", img_folder, fname_prefix, fig=fig_y, custom_lab=lab, ratio=(True if ratio_id is not None else False), colour_skip=colour_skip, fancy=fancy)
		colour_skip += 1
	ax_x = fig_x.axes[0]
	box = ax_x.get_position()
	ax_x.set_position([box.x0, box.y0 + box.height * 0.15,
				box.width, box.height * 0.85])
	legend_x = ax_x.legend(
		fontsize=8, title_fontsize=8,
		loc='upper center', bbox_to_anchor=(0.5, -0.15),
		fancybox=True, shadow=True, title=('' if leg_title is None else leg_title)
	)
	ax_y = fig_y.axes[0]
	box = ax_y.get_position()
	ax_y.set_position([box.x0, box.y0 + box.height * 0.15,
				box.width, box.height * 0.85])
	legend_y = ax_y.legend(
		fontsize=8, title_fontsize=8,
		loc='upper center', bbox_to_anchor=(0.5, -0.15),
		fancybox=True, shadow=True, title=('' if leg_title is None else leg_title)
	)
	ax_x.grid(True, zorder=0)
	ax_y.grid(True, zorder=0)
	for frmt in frmts:
		print(img_folder+fname_prefix+('_ratio' if ratio_id is not None else '')+f"_1Dscan_MX_OR.{frmt}")
		fig_x.savefig(img_folder+fname_prefix+('_ratio' if ratio_id is not None else '')+f"_1Dscan_MX_OR.{frmt}", bbox_inches='tight', dpi=500)
		fig_y.savefig(img_folder+fname_prefix+('_ratio' if ratio_id is not None else '')+f"_1Dscan_MY_OR.{frmt}", bbox_inches='tight', dpi=500)
	if y_cut is not None: # y_cut should be a tuple
		if len(y_cut) == 4:
			ymin_x, ymax_x, ymin_y, ymax_y = y_cut
		elif len(y_cut) == 2:
			ymin_x, ymax_x, ymin_y, ymax_y = y_cut[0], y_cut[1], y_cut[0], y_cut[1]
		ax_x.set_ylim(ymin_x, ymax_x)
		ax_y.set_ylim(ymin_y, ymax_y)
		for frmt in frmts:
			fig_x.savefig(img_folder+fname_prefix+('_ratio' if ratio_id is not None else '')+f"_cut_1Dscan_MX_OR.{frmt}", bbox_inches='tight', dpi=500)
			fig_y.savefig(img_folder+fname_prefix+('_ratio' if ratio_id is not None else '')+f"_cut_1Dscan_MY_OR.{frmt}", bbox_inches='tight', dpi=500)


def plot_2d_sens_scan(df_bkg, df_sig, trg_set, title, logx=False, fancy=False):
	'''
	Plot MX vs MY scan of sensitivity
	'''

	combs = [f"MX_{int(r['MX'])}_MY_{int(r['MY'])}" for _, r in df_sig.iterrows()]

	qcd_row = df_bkg.iloc[0]
	for j, row in df_sig.iterrows():
		fig, ax = plt.subplots()
		ax.set_axisbelow(True)
		ax.grid()
		ax.set_prop_cycle(color=custom_colours)
		for i, ylab in enumerate(trg_set):
			lab = map_id_to_lab[ylab]
			mlab = lab[5:] if lab.startswith("OR   ") else lab
			scatt = ax.scatter(qcd_row['eff_'+ylab], row['eff_'+ylab], zorder=1, label=lab, marker=('o' if mlab not in map_trg_to_marker.keys() else map_trg_to_marker[mlab]))
			if round(row[ylab], 3)==1.:
				continue
			text = ax.annotate(
				f"({qcd_row['eff_'+ylab]:.3f},\n {row['eff_'+ylab]:.3f})",
				(qcd_row['eff_'+ylab], row['eff_'+ylab]),
				textcoords="offset points",
				xytext=(5, 5),
				fontsize=8,
				color=scatt.get_facecolor(),
			)
			text.set_path_effects([PathEffects.withStroke(linewidth=0.5, foreground='black')]) ###

		if logx:
			ax.set_xscale('log')
		ax.set_xlabel(r'$\epsilon_B$'+('' if not logx else ' (log)'))
		ax.set_ylabel(r' $\epsilon_S$')
		ax.set_xlim(0., 1.)
		ax.set_ylim(0., 1.)
		m_comb_for_title = rf"$M_X={combs[j].split('_')[1]}$ GeV, $M_Y={combs[j].split('_')[3]}$ GeV"
		ax.set_title(title+' '+m_comb_for_title)
		ax.set_box_aspect(1)
		for frmt in frmts+['png']:
			if frmt == 'pdf':
				ax.legend(loc='upper center', fancybox=True, shadow=True, fontsize=8, bbox_to_anchor=(0.5, -0.15))
			fig.savefig(img_folder+f"sens_2Dscan_{combs[j]}"+('' if not logx else'_logx')+f".{frmt}", bbox_inches='tight', dpi=500)



parser = argparse.ArgumentParser(description="Read CSV Efficiencies records and make plots.")
group_trggrp = parser.add_mutually_exclusive_group(required=False)
group_plt = parser.add_mutually_exclusive_group(required=False)
group_trggrp.add_argument("--all", action="store_true", help="Plot everything.")
group_trggrp.add_argument("--really_all", action="store_true", help="Plot really everything.")
group_trggrp.add_argument("--t",  type=str, help="Trigger set to use.") # ground, VBF_std, VBF_park, VBF_std_VBF_park, ground_VBF_std_VBF_park
group_trggrp.add_argument("--gv", action="store_true", help="Plot the ggF vs one of the VBF triggers.")
group_trggrp.add_argument("--gv2", action="store_true", help="Plot the ggF vs one of the VBF triggers. (2)")
group_trggrp.add_argument("--pr", action="store_true", help="Plot the proper trigger groups comparison.")
group_trggrp.add_argument("--sens", action="store_true", help="Plot the efficiencies of triggers considering also BKG samples.")
group_plt.add_argument("--fancy", action="store_true", help="Plot for the presentation.")
args = parser.parse_args()


print(args)

if (args.all or args.t is not None) or args.really_all: # if you specify which set to use 
	if args.t is not None:
		print("t plotting\n\n")
		ids = [args.t]
	else:
		print("all plotting\n\n")
		ids = [s.split('_', 2)[-1].split('.')[0] for s in os.listdir('csvs') if 'bkg' not in s]

	for id in ids:
		print(id)
		img_folder = f'imgs/{id}/'
		if not os.path.exists(img_folder):
			os.makedirs(img_folder)

		scan_x, scan_y, df, trigger_names = preprocess(id)

		skip_len = 3+2*len(trigger_names)+2
		indiv_trgs = df.columns[skip_len:skip_len+len(trigger_names)]
		_ = plot_1d_scan(scan_x, "MX", indiv_trgs, r"Individual Efficiency - $M_X$ scan (fixed $M_Y=125$ GeV)", img_folder, id, fancy=args.fancy)
		_ = plot_1d_scan(scan_y, "MY", indiv_trgs, r"Individual Efficiency - $M_Y$ scan (fixed $M_X=1000$ GeV)", img_folder, id, fancy=args.fancy)
		excl_trgs = df.columns[skip_len+len(trigger_names):skip_len+2*len(trigger_names)]
		_ = plot_1d_scan(scan_x, "MX", excl_trgs, r"Exclusive Efficiency - $M_X$ scan (fixed $M_Y=125$ GeV)", img_folder, id, fancy=args.fancy)
		_ = plot_1d_scan(scan_y, "MY", excl_trgs, r"Exclusive Efficiency - $M_Y$ scan (fixed $M_X=1000$ GeV)", img_folder, id, fancy=args.fancy)

		_ = plot_1d_scan(scan_x, "MX", ["eff_any"], r"OR Efficiency - $M_X$ scan (fixed $M_Y=125$ GeV)", img_folder, id, fancy=args.fancy)
		_ = plot_1d_scan(scan_y, "MY", ["eff_any"], r"OR Efficiency - $M_Y$ scan (fixed $M_X=1000$ GeV)", img_folder, id, fancy=args.fancy)
		_ = plot_1d_scan(scan_x, "MX", ["eff_all"], r"AND Efficiency - $M_X$ scan (fixed $M_Y=125$ GeV)", img_folder, id, fancy=args.fancy)
		_ = plot_1d_scan(scan_y, "MY", ["eff_all"], r"AND Efficiency - $M_Y$ scan (fixed $M_X=1000$ GeV)", img_folder, id, fancy=args.fancy)


		plot_2d_scan(df, 'eff_any', "OR Efficiency - 2D scan", img_folder, id, fancy=args.fancy)
		plot_2d_scan(df, 'eff_all', "AND Efficiency - 2D scan", img_folder, id, fancy=args.fancy)

		# de-comment for interactive 3d plot
		#plt.show(block=True)
		#import matplotlib
		#matplotlib.use('TkAgg')

		plot_2d_3d_scan(df, "3d plot of efficiency", img_folder, id, fancy=args.fancy)
		plot_2_contribs(df, "OR Efficiency - $M_X$ scan (fixed $M_Y=125$ GeV)"+'\n'+'Individual contributions', img_folder, id, 'x', fancy=args.fancy)
		plot_2_contribs(df, "OR Efficiency - $M_Y$ scan (fixed $M_X=1000$ GeV)"+'\n'+'Individual contributions', img_folder, id, 'y', fancy=args.fancy)
		print()

if (args.gv) or args.really_all:  # if you want to compare the combinations "Run3 Inclusive OR the other triggers"
	print("gv plotting\n\n")
	img_folder = f'imgs/gv/'
	if not os.path.exists(img_folder):
		os.makedirs(img_folder)
	ids = [s.split('_', 2)[-1].split('.')[0] for s in os.listdir('csvs')]
	ids = [id for id in ids if bool(re.search(r"001_\d{3}", id))]
	labels = ["OR   "+map_id_to_lab[trigger_names_ids[int(id.split("_")[2])]] for id in ids]
	ids.insert(0, 'tr_001')
	labels.insert(0, map_id_to_lab[trigger_names_ids[1]]+' ')
	plot_1d_scan_comparison(img_folder, ids, labels, colour_skip=2-1, y_cut=(0.4, 1.), fancy=args.fancy)

if (args.gv2) or args.really_all: # if you want to compare the combinations "Run3 Inclusive OR VBF specific triggers"
	print("gv2 plotting\n\n")
	img_folder = f'imgs/gv2/'
	if not os.path.exists(img_folder):
		os.makedirs(img_folder)
	ids = [s.split('_', 2)[-1].split('.')[0] for s in os.listdir('csvs')]
	ids = [id for id in ids if bool(re.search(r"001_\d{3}", id))]
	labels = ["OR   "+map_id_to_lab[trigger_names_ids[int(id.split("_")[2])]] for id in ids]
	ids.insert(0, 'tr_001')
	labels.insert(0, map_id_to_lab[trigger_names_ids[1]]+' ')
	plot_1d_scan_comparison(img_folder, ids, labels, colour_skip=2-1, y_cut=(0.4, 1.), fancy=args.fancy, fname_prefix='gv2', skip_ids=[1, 2, 3])

if (args.pr) or args.really_all: # compare selected combinations: run2 analogous, run3 inclusive, OR between the two SS, OR between the three PS, Run3 Analogous OR first SS, Run3 Analogous OR first PS
	print("pr plotting\n\n")
	img_folder = f"imgs/pr/"
	if not os.path.exists(img_folder):
		os.makedirs(img_folder)
	ids = ["tr_000", "tr_001", "VBF_std", "VBF_park"] + ["tr_002", "tr_004"]
	labels = [trigger_labels[0], trigger_labels[1], "VBF Standard Stream", "VBF Parking Stream"] + [f"{trigger_labels[1]} OR {trigger_labels[2]}", f"{trigger_labels[1]} OR {trigger_labels[4]}"]
	plot_1d_scan_comparison(img_folder, ids, labels, fancy=args.fancy)
	plot_1d_scan_comparison(img_folder, ids, labels, ratio_id='tr_000', fancy=args.fancy)

if (args.sens) or args.really_all: # compare sensibilities
	print("sens plotting\n\n")
	img_folder = f"imgs/sens/"
	if not os.path.exists(img_folder):
		os.makedirs(img_folder)
	scan_x, scan_y, df_sig, df_bkg, trigger_names = preprocess_sens()

	plot_1d_scan(scan_x, "MX", ['sens_'+t for t in trigger_names], r"Sensitivity - $M_X$ scan (fixed $M_Y=125$ GeV)", img_folder, 'sens', sens=True, fancy=False)
	plot_1d_scan(scan_y, "MY", ['sens_'+t for t in trigger_names], r"Sensitivity - $M_Y$ scan (fixed $M_X=1000$ GeV)", img_folder, 'sens', sens=True, fancy=False)

	plot_2d_sens_scan(df_bkg, df_sig, trigger_names, '', fancy=args.fancy)
	plot_2d_sens_scan(df_bkg, df_sig, trigger_names, '', logx=True, fancy=args.fancy)

import os
import numpy as np
import awkward as ak
import glob
import argparse
from tqdm import tqdm
import ROOT
from itertools import combinations
import csv
import sys
import json
import pandas as pd
from pathlib import Path
import re
import torch
from var_funcs import *
from dnn.models.dnn import *
from paths import *


def get_vbf_pair(DeltaRs_gamma, DeltaRs_delta, nJet, jet_nob_partFlavs, verbose=False):
	"""
	Get the true vbf pair
	"""
	ok_pair = []
	for i in range(nJet-4):
		for j in range(i+1, nJet-4):
			i_gamma = True if DeltaRs_gamma[i] < DeltaRs_delta[i] else False
			j_delta = True if DeltaRs_delta[j] < DeltaRs_gamma[j] else False
			DeltaR_gamma = DeltaRs_gamma[i] if i_gamma else DeltaRs_delta[i]
			DeltaR_delta = DeltaRs_delta[j] if j_delta else DeltaRs_gamma[j]
			check = (DeltaR_gamma < 0.4) & (DeltaR_delta < 0.4)
			check2 = ((min(DeltaRs_gamma[i], DeltaRs_delta[i]) < 0.4) & (min(DeltaRs_gamma[j], DeltaRs_delta[j]) < 0.4))
			if check != check2: # just to check if the explcit condition matches the chain
				print('Error: checks do not match')
			if check:
				ok_pair.append({'pair': (i, j), 'DeltaR_gamma': DeltaR_gamma, 'DeltaR_delta': DeltaR_delta, 'ambiguous': (1 if i_gamma != j_delta else 0)})
				if i_gamma != j_delta and verbose:
						print(f'Error: the same parton was associated to two different jets ({i}, {j})')

	if len(ok_pair)>0:
		ordered = sorted(
			ok_pair,
			key=lambda x: x['DeltaR_gamma']**2 + x['DeltaR_delta']**2
		)
		while len(ordered)>0 and ordered[0]['ambiguous']==1:
			i, j = ordered[0]['pair']
			if (abs(jet_nob_partFlavs[i])>=5) or (abs(jet_nob_partFlavs[j])>=5): # discard it if it's of bottom nature or more 
				_ = ordered.pop(0)
			else:
				ok_pair = [ordered[0]]
				ordered = []
		if len(ordered)>0:
			ok_pair = [ordered[0]]

	if len(ok_pair)>0: # ambiguity check
		if ok_pair[0]['ambiguous'] == 1 and verbose:
			print('Final chosen pair is ambiguous')

	vbf_pair = ok_pair[0]['pair'] if len(ok_pair)>0 else None
	if len(ok_pair)>0:
		if vbf_pair[0] < vbf_pair[1]:
			vbf_dists = {'DeltaR_1': ok_pair[0]['DeltaR_gamma'], 'DeltaR_2': ok_pair[0]['DeltaR_delta']} 
		else:
			vbf_dists = {'DeltaR_1': ok_pair[0]['DeltaR_delta'], 'DeltaR_2': ok_pair[0]['DeltaR_gamma']}
	else:
		vbf_dists = None

	return vbf_pair, vbf_dists


def get_vars(pairs, jet_ps_nob, sample, unq_evt_id, vbf_pair, p_X, b_jets, Jet_btagUParTAK4QvG, Jet_btagUParTAK4B, Jet_btagUParTAK4SvUDG):
	"""
	Compute the features used by the NN
	Functions are defined in var_funcs.py
	"""
	id1 = np.array(pairs)[:, 0]
	id2 = np.array(pairs)[:, 1]
	jet1 = np.array(jet_ps_nob)[id1]
	jet2 = np.array(jet_ps_nob)[id2]
	results = {
		'MX'		: ((int(sample.split('_')[1]) if sample != 'QCD' else 0) * np.ones(len(id1))).astype(int),
		'MY'		: ((int(sample.split('_')[3]) if sample != 'QCD' else 0) * np.ones(len(id1))).astype(int),
		'event'     : (unq_evt_id * np.ones(len(id1))).astype(int),
		'id1'       : id1,
		'id2'       : id2,
		'is_vbf'    : (((id1 == vbf_pair[0]) & (id2 == vbf_pair[1])).astype(int)) if vbf_pair else np.zeros(len(id1)).astype(int),
		'mjj'       : np.vectorize(var_mjj)(jet1, jet2),
		'pTjj'      : np.vectorize(var_pTjj)(jet1, jet2),
		'dEta'      : np.vectorize(var_dEta)(jet1, jet2),
		'pT_asymm'  : np.vectorize(var_pT_asymm)(jet1, jet2),
		'rap_prod'  : np.vectorize(var_rap_prod)(jet1, jet2),
		'jet_props' : np.vectorize(lambda i1, i2: var_jet_props(jet_ps_nob, Jet_btagUParTAK4QvG, Jet_btagUParTAK4B, Jet_btagUParTAK4SvUDG, i1, i2))(id1, id2),
		'zepp'      : np.vectorize(var_zepp)(p_X, jet1, jet2),
		'rap_gap'   : np.vectorize(lambda j1, j2: var_rap_gap(b_jets, j1, j2))(jet1, jet2),
		'jet_iso'   : np.vectorize(lambda i1, i2: var_jet_iso(jet_ps_nob, i1, i2))(id1, id2),
		'ang_seps'  : np.vectorize(lambda j1, j2: var_ang_seps(b_jets, j1, j2))(jet1, jet2),
		'cos_theta' : np.vectorize(var_cos_theta)(p_X, jet1, jet2),
		'theta' 	: np.vectorize(var_theta)(p_X, jet1, jet2),
		'pT_balance': np.vectorize(var_pT_balance)(p_X, jet1, jet2)
	}

	# flatten the results
	jet_props = results.pop("jet_props")
	jet_iso = results.pop("jet_iso")
	ang_seps = results.pop("ang_seps")
	flattened = {}
	for pair in jet_props:
		jets = list(pair.values())
		for i, jet in enumerate(jets, start=1):
			for prop, value in jet.items():
				key = f"jet_props_{i}_{prop}"
				flattened.setdefault(key, []).append(float(value) if value is not None else np.nan)
	for pair in jet_iso:
		jets = list(pair.values())
		for i, jet in enumerate(jets, start=1):
			for prop, value in jet.items():
				key = f"jet_iso_{i}_{prop}"
				flattened.setdefault(key, []).append(float(value) if value is not None else np.nan)
	for jet_name in list(ang_seps[0].keys()):
		for prop in ['DeltaR', 'DeltaPhi', 'DeltaRap']:
			key = f"ang_seps_{jet_name}_{prop}"
			flattened[key] = np.array([
				[sep[prop] for sep in event[jet_name]]
				for event in ang_seps
			], dtype=object)
			max_len = 4
			flattened[key] = np.array([
				[sep[prop] for sep in event[jet_name]][:max_len]
				+ [np.nan] * max(0, max_len - len(event[jet_name]))
				for event in ang_seps
			], dtype=float)
	flattened = {
		key: np.array(values)
		for key, values in flattened.items()
	}
	results.update(flattened)
	results_flat = {}
	for name, arr in results.items():
		arr = np.asarray(arr)
		if arr.ndim == 1:
			results_flat[name] = arr
		elif arr.ndim == 2:
			for i in range(arr.shape[1]):
				results_flat[f"{name}_{i}"] = arr[:, i]
		else:
			raise ValueError(f"{name} has unsupported shape {arr.shape}")

	return results_flat


# Load DNN definitions
DNN_FUNCTIONS = {
    int(name.split("_")[1]): func
    for name, func in globals().items()
    if name.startswith("dnn_") and callable(func)
}


def get_model(model_path, id):
	"""
	Load the NN models
	"""
	dims = [
		v.shape 
		for k, v in torch.load(model_path).items() 
		if k.endswith(".weight") and v.ndim == 2
	]
	arch = [dims[0][1]] + [x[0] for x in dims]
	inp_dim = arch[0]
	hid_dims = arch[1:-1]
	net = DNN_FUNCTIONS[id](inp_dim, hid_dims)
	net.load_state_dict(torch.load(model_path, weights_only=True))
	return net


# Ignore ROOT and numpy warnings and set verbose
ROOT.gErrorIgnoreLevel = ROOT.kError
np.seterr(invalid="ignore")
verbose = False

# List of trigger names
trigger_names = [
		# Baseline triggers
		"HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5", # Analogous to Run 2   # 000
		"HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55", # Run 3 Inclusive                          # 001
		# Standard stream
		"HLT_VBF_DiPFJet125_45_Mjj1050",                                                        # 002
		"HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60",                                                # 003
		# Parking stream
		"HLT_QuadPFJet103_88_75_15",                                                            # 004
		"HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1",                                    # 005
		"HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2",                                          # 006
]

# Mass combinations
m_combs = ['MX_1000_MY_125',
		 'MX_500_MY_125', 'MX_700_MY_125', 'MX_1400_MY_125', 'MX_2000_MY_125',
		 'MX_1000_MY_90', 'MX_1000_MY_200', 'MX_1000_MY_400', 'MX_1000_MY_600']
# VBF files
m_combs_paths = {comb: VBF_PATH.format(comb=comb) for comb in m_combs}

# ggF files
ggf_m_combs_paths = {comb: GGF_PATH.format(comb=comb.replace('_', '-')) for comb in m_combs}

# backgrounds
bkgs = ['QCD']
bkgs_paths = {'QCD': QCD_PATH}

parser = argparse.ArgumentParser(description="Read NanoAOD files and calculate trigger efficiencies.")
parser.add_argument("--VBF", action="store_true", help="Whether the mass scan concerns VBF samples or not.")
parser.add_argument("--ggF", action="store_true", help="Whether the mass scan concerns ggF samples or not.")
parser.add_argument("--sample", type=str, help="Sample to read (for the moment: QCD (bkg)")
parser.add_argument("--ow", action="store_true", help="Overwrite already existent files.")
parser.add_argument("--mjj", action="store_true", help="Choose mjj-based selection of the VBF Jet pair.")
parser.add_argument("--dnn", action="store_true", help="Choose DNN-based selection of the VBF Jet pair.")
parser.add_argument("--m_id", type=int, help="(integer) Choose DNN model type for the selection of the VBF Jet pair.")
parser.add_argument("--d_id", type=str, help="(str) Choose dataset type for the DNN-selection of the VBF Jet pair.")
parser.add_argument("--debug", action="store_true", help="Debug: runs script on just a few events.")
args = parser.parse_args()


has_VBF = args.VBF
has_ggF = args.ggF
has_sample = args.sample is not None


choices = [has_VBF, has_ggF, has_sample]

if sum(choices) != 1:
	parser.error(
			"You must specify exactly one of: "
			"--VBF, --ggF, or --sample."
	)

if has_sample and args.sample not in bkgs:
	parser.error(
			"You must select one amongst the available samples."
	)

# Select samples to consider
if has_sample:
	samples = [args.sample]
	flab = args.sample
elif has_ggF or has_VBF:
	samples = m_combs
	if has_ggF and not has_VBF:
		flab = 'ggF'
	else:
		flab = 'VBF'


if sum([args.mjj, args.dnn]) != 1:
	parser.error("You must specify either --mjj or --dnn.")
if args.dnn and args.m_id is None:
	parser.error("You must specify the model type --m_id when using the option --dnn.")
if args.dnn and args.d_id is None:
	parser.error("You must specify the dataset type --d_id when using the option --dnn.")
if args.dnn:
	root_dir = 'dnn_sel/'
	model_list = [
		file
		for directory in Path('dnn/results/').rglob("*")
		if directory.is_dir() and f"{args.m_id:03d}" in directory.name
		for file in directory.glob("*.pt")
	]
	if args.m_id == 1:
		model_dict = {
			m.group(): filename
			for filename in model_list
			if (m := re.search(r"MX_\d+_MY_\d+", str(filename)))
		}
	dataset_vars_dict = {
		m.group(): file
		for directory in Path("dnn/datasets/").rglob("*")
		if directory.is_dir() and str(args.d_id) in directory.name
		for file in directory.glob("*.txt")
		if (m := re.search(r"MX_\d+_MY_\d+", str(file))) and ('used' in str(file))
	}
	dataset_logvars_dict = {
		m.group(): file
		for directory in Path("dnn/datasets/").rglob("*")
		if directory.is_dir() and str(args.d_id) in directory.name
		for file in directory.glob("*.json")
		if (m := re.search(r"MX_\d+_MY_\d+", str(file))) and ('log' in str(file))
	}
	dataset_absvars_dict = {
		m.group(): file
		for directory in Path("dnn/datasets/").rglob("*")
		if directory.is_dir() and str(args.d_id) in directory.name
		for file in directory.glob("*.txt")
		if (m := re.search(r"MX_\d+_MY_\d+", str(file))) and ('abs' in str(file))
	}
	norm_stats_dict = {
		m.group(): file
		for directory in Path("dnn/datasets/").rglob("*")
		if directory.is_dir() and str(args.d_id) in directory.name
		for file in directory.glob("*.json")
		if (m := re.search(r"MX_\d+_MY_\d+", str(file))) and ('trainval' in str(file))
	}
	print('vars', dataset_vars_dict)
	print()
	print('logvars', dataset_logvars_dict)
	print()
	print('absvars', dataset_absvars_dict)
	print()
	print('norm stats', norm_stats_dict)
	print()
elif args.mjj:
	root_dir = 'mjj_sel/'





out_dir_csv = root_dir+("debug/" if args.debug else "")+f"csvs/"
os.makedirs(out_dir_csv, exist_ok=True)
csv_filename = out_dir_csv+f"{flab}.csv"
csv_fieldnames = ["sample", "tot_n_evts", "trg_cut", "num_jet_cut", "loose_b_wp", "vbf_cut", "purity_test", "purity_denominator"]
try:
	file = open(csv_filename, mode="w" if args.ow else "x", newline="")
	writer = csv.writer(file)
	writer.writerow(csv_fieldnames)
except FileExistsError:
	if not args.ow:
		print(f"Error: {csv_filename} already exists. Exiting program.")
		raise SystemExit


if args.debug:
	samples = [samples[0]]
for sample in samples:

	print(f"--- {sample} ---")

	orig_dir_npz = root_dir+("debug/" if args.debug else "")+"npzs/"
	out_dir_npz = orig_dir_npz+f"npzs_{flab}_{sample}/"
	os.makedirs(out_dir_npz, exist_ok=True)


	if has_VBF:
		file_pattern = m_combs_paths[sample]
	elif has_ggF:
		file_pattern = ggf_m_combs_paths[sample]
	elif sample in bkgs:
		file_pattern = bkgs_paths[sample]
	print(file_pattern)
	file_paths = glob.glob(file_pattern)

	# Variable initializations
	total_events = 0
	total_passed_any = 0
	jet_num_cut_passed = 0
	looseWP_cut_passed = 0
	vbf_cut_passed = 0
	purity_test = 0
	purity_denominator = 0
	Hts_pre = []
	mjjs_pre = []
	detas_pre = []
	trg_dicts = {trgname: {'ht': [], 'mjj': [], 'deta': []} for trgname in trigger_names}
	mjjs_blwp = []
	detas_blwp = []
	mjjs_blwp_pure = []
	detas_blwp_pure = []
	mjjs_vbfCut = []
	detas_vbfCut = []
	mjjs_vbfCut_pure = []
	detas_vbfCut_pure = []
	jet_blwp_part_DeltaR_1_s = []
	jet_blwp_part_DeltaR_2_s = []
	jet_blwp_part_flav_1_s = []
	jet_blwp_part_flav_2_s = []
	jet_vbfCut_part_DeltaR_1_s = []
	jet_vbfCut_part_DeltaR_2_s = []
	jet_vbfCut_part_flav_1_s = []
	jet_vbfCut_part_flav_2_s = []
	sig_count = []
	bkg_count = []
	mjj_dnn_match = []
	is_there_a_vbf_pair = []
	dnn_eff = []
	bjets_rows = []

	# Filepaths definitions
	pre_npz_name = out_dir_npz+'pre_trg.npz'
	if os.path.exists(pre_npz_name) and not args.ow:
		print(f"Error: {pre_npz_name} file already exists, exiting the program.")
		sys.exit(1)
	trg_npz_names = {trg_name: out_dir_npz+trg_name+'.npz' for trg_name in trigger_names}
	for name in [pre_npz_name]+list(trg_npz_names.values()):
		if os.path.exists(name) and not args.ow:
			print(f"Error: {name} file already exists, exiting the program.")
			sys.exit(1)

	blwp_npz_name = out_dir_npz+'blwp_evts.npz'
	if os.path.exists(blwp_npz_name) and not args.ow:
		print(f"Error: {blwp_npz_name} file already exists, exiting the program.")
		sys.exit(1)
	blwp_pure_npz_name = out_dir_npz+'blwp_pure_evts.npz'
	if os.path.exists(blwp_pure_npz_name) and not args.ow:
		print(f"Error: {blwp_pure_npz_name} file already exists, exiting the program.")
		sys.exit(1)

	vbfCut_npz_name = out_dir_npz+'vbfCut_evts.npz'
	if os.path.exists(vbfCut_npz_name) and not args.ow:
		print(f"Error: {vbfCut_npz_name} file already exists, exiting the program.")
		sys.exit(1)
	vbfCut_pure_npz_name = out_dir_npz+'vbfCut_pure_evts.npz'
	if os.path.exists(vbfCut_pure_npz_name) and not args.ow:
		print(f"Error: {vbfCut_pure_npz_name} file already exists, exiting the program.")
		sys.exit(1)

	jetMatches_blwp_npz_fname = out_dir_npz+"JetMatches_blwp.npz"
	if os.path.exists(jetMatches_blwp_npz_fname) and not args.ow:
		print(f"Error: {jetMatches_blwp_npz_fname} file already exists, exiting the program.")
		sys.exit(1)
	jetMatches_vbfCut_npz_fname = out_dir_npz+"JetMatches_vbfCut.npz"
	if os.path.exists(jetMatches_vbfCut_npz_fname) and not args.ow:
		print(f"Error: {jetMatches_vbfCut_npz_fname} file already exists, exiting the program.")
		sys.exit(1)


	out_dir_parquet = root_dir+("debug/" if args.debug else "")+'vbf_parquets/'
	os.makedirs(out_dir_parquet, exist_ok=True)
	parquet_fname = out_dir_parquet+f"vars_{sample}.parquet"
	if os.path.exists(parquet_fname) and not args.ow:
		print(f"Error: {parquet_fname} file already exists, exiting the program.")
		sys.exit(1)

	if args.dnn:
		out_dir_purpar = root_dir+("debug/" if args.debug else "")+'vbf_purity_parquets/'
		os.makedirs(out_dir_purpar, exist_ok=True)
		purpar_fname = out_dir_purpar+f"purity_{sample}.parquet"
		if os.path.exists(purpar_fname) and not args.ow:
			print(f"Error: {purpar_fname} file already exists, exiting the program.")
			sys.exit(1)
		out_dir_dnn_eff = root_dir+("debug/" if args.debug else "")+'dnn_eff/'
		os.makedirs(out_dir_dnn_eff, exist_ok=True)
		dnn_eff_fname = out_dir_dnn_eff+f"dnn_eff_{flab}_{sample}.parquet"
		if os.path.exists(dnn_eff_fname) and not args.ow:
			print(f"Error: {dnn_eff_fname} file already exists, exiting the program.")
			sys.exit(1)

	out_dir_bjets = root_dir+("debug/" if args.debug else "")+'bjets/'
	os.makedirs(out_dir_bjets, exist_ok=True)
	bjets_fname = out_dir_bjets+f"bjets_{flab}_{sample}.parquet"
	if os.path.exists(bjets_fname) and not args.ow:
		print(f"Error: {bjets_fname} file already exists, exiting the program.")
		sys.exit(1)


	tnames = ['hlt'+t[3:] for t in trigger_names] if sample == 'QCD' else trigger_names


	# Full cycle for the sample
	unq_evt_id = -1
	paths_evt_ids = {}
	df = None
	pur_rows = []
	if not ((os.path.exists(pre_npz_name) and all(os.path.exists(p) for p in trg_npz_names.values())) and not args.ow):
		if args.debug:
			file_paths = file_paths[:10] ## just for testing
		for path in tqdm(file_paths, desc="Processing NanoAODs", unit="file"):
			paths_evt_ids[path] = []
			with ROOT.TFile.Open(path, "READ") as file:
				tree = file.Get("Events")
				for j, event in enumerate(tree):
					unq_evt_id += 1
					paths_evt_ids[path].append(unq_evt_id)

					if args.debug and j >= 200:
						continue

					passed = True # Flag for all conditions in cut flows conditions, minus VBF
					passed_vbf_cut = True # Flag for VBF cuts passing

					total_events += 1

					jet_ps = []
					if sample == 'QCD': # different naming in QCD files
						nJet = event.nPFJetAK4
						jet_pts = np.asarray(event.PFJetAK4_pt)
						jet_etas = np.asarray(event.PFJetAK4_eta)
						jet_phis = np.asarray(event.PFJetAK4_phi)
						jet_masss = np.asarray(event.PFJetAK4_mass)
					else:
						nJet = event.nJet
						jet_pts = np.asarray(event.Jet_pt)
						jet_etas = np.asarray(event.Jet_eta)
						jet_phis = np.asarray(event.Jet_phi)
						jet_masss = np.asarray(event.Jet_mass)


					jet_pts = np.where(jet_pts < 15, np.nan, jet_pts)
					jet_etas = np.where(np.abs(jet_etas) > 4.7, np.nan, jet_etas)


					for pt, eta, phi, mass in zip(jet_pts, jet_etas, jet_phis, jet_masss):
						p = ROOT.TLorentzVector()
						p.SetPtEtaPhiM(pt, eta, phi, mass)
						jet_ps.append(p)


					if len(jet_pts) <= 1:
						continue

					if getattr(event, f"{tnames[1]}") | getattr(event, f"{tnames[5]}"): ###
						total_passed_any += 1
					else:
						passed = False

					if len(jet_ps) >= 6 and passed:
						jet_num_cut_passed += 1
					else:
						passed = False

					Ht = np.sum(jet_pts[:6], axis=0)

					# UParT algo scores
					Jet_btagUParTAK4B = np.asarray(event.Jet_btagUParTAK4B) if sample != 'QCD' else np.asarray(event.PFJetAK4_btagUParTAK4B)
					Jet_btagUParTAK4QvG = np.asarray(event.Jet_btagUParTAK4QvG) if sample != 'QCD' else np.nan * np.ones(len(Jet_btagUParTAK4B))
					Jet_btagUParTAK4SvUDG = np.asarray(event.Jet_btagUParTAK4SvUDG) if sample != 'QCD' else np.nan * np.ones(len(Jet_btagUParTAK4B))

					# Loose b-tagging WP
					if sum(Jet_btagUParTAK4B > 0.0246) >= 4 and passed:
						looseWP_cut_passed +=1
					else:
						passed = False

					bJetsIds = sorted(range(len(Jet_btagUParTAK4B)), key=lambda i: Jet_btagUParTAK4B[i], reverse=True)[:4]
					bJetsScores = [score for _, score in sorted(enumerate(Jet_btagUParTAK4B), key=lambda x: x[1], reverse=True)[:4]]
					b_jets = list(np.array(jet_ps)[bJetsIds])
					jet_ps_nob = list(np.array(jet_ps)[~np.isin(np.arange(len(jet_ps)), bJetsIds)])
					if any(x is y for x in b_jets for y in jet_ps_nob):
						print('Error: b_jet list and non b jet list are sharing objects!')

					p_X = sum(b_jets, ROOT.TLorentzVector())

					# All possible jet pairs in the event, b-jets excluded
					pairs = list(combinations(np.arange(len(jet_ps_nob)), 2))

					if len(pairs) == 0:
						if passed:
							print("Error: zero pairs left after removing 4b")
						continue

					# Gen Partons Four-Vectors
					part_ps = []
					part_pts = np.asarray(event.GenPart_pt)
					part_etas = np.asarray(event.GenPart_eta)
					part_phis = np.asarray(event.GenPart_phi)
					part_masss = np.asarray(event.GenPart_mass)
					for pt, eta, phi, mass in zip(part_pts, part_etas, part_phis, part_masss):
						p = ROOT.TLorentzVector()
						p.SetPtEtaPhiM(pt, eta, phi, mass)
						part_ps.append(p)

					# Gen Partons Four-Vectors
					GenPart_pdgId = np.asarray(event.GenPart_pdgId)
					GenPart_status = np.asarray(event.GenPart_status)

					# Partons corresponding to the VBF Jets
					vbf_part_cond = (
						(abs(GenPart_pdgId)>=1)
						& (abs(GenPart_pdgId)<5)
						& (GenPart_status==23)
					)

					if sample != 'QCD':
						GenPart_statusFlags = np.asarray(event.GenPart_statusFlags)
						vbf_part_cond_old = vbf_part_cond
						vbf_part_cond = (vbf_part_cond & ((GenPart_statusFlags & 128)!=0))
						if not np.array_equal(vbf_part_cond, vbf_part_cond_old):
							print('Error: "((GenPart_statusFlags & 128)!=0) was needed')
							sys.exit(1)

					vbf_part_ids = np.where(vbf_part_cond)[0]

					# Jet Parton Flavour
					jet_partFlavs = np.asarray(event.Jet_partonFlavour) if sample != 'QCD' else None
					jet_nob_partFlavs = list(np.array(jet_partFlavs)[~np.isin(np.arange(len(jet_partFlavs)), bJetsIds)]) if sample != 'QCD' else None


					if len(vbf_part_ids) <= 1 or sample == 'QCD' or has_ggF:
						if verbose:
							print(f'Error: one or less VBF partons were found ({vbf_part_ids})')
						vbf_pair = False
					else:
						# Distances of jets with VBF-partons
						part_gamma, part_delta = part_ps[vbf_part_ids[0]], part_ps[vbf_part_ids[1]]
						DeltaRs_gamma = [part_gamma.DeltaR(jet_ps_nob[i], True) for i in range(nJet-4)]
						DeltaRs_delta = [part_delta.DeltaR(jet_ps_nob[i], True) for i in range(nJet-4)]

						# Find the VBF pair
						vbf_pair, vbf_dists = get_vbf_pair(DeltaRs_gamma, DeltaRs_delta, nJet, jet_nob_partFlavs)

					is_there_a_vbf_pair.append(True if vbf_pair is False or vbf_pair is not None else False)


					# Compute all the variables relative to the jets pairs
					results_flat = get_vars(pairs, jet_ps_nob, sample, unq_evt_id, vbf_pair, p_X, b_jets,
											Jet_btagUParTAK4B, Jet_btagUParTAK4QvG, Jet_btagUParTAK4SvUDG)
					# Save
					df_res = pd.DataFrame(results_flat)

					if passed and has_VBF:
						if df is not None:
							df = pd.concat([df, df_res], ignore_index=True)
						else:
							df = df_res


					if args.mjj:
						mjjs = {pair: (jet_ps_nob[pair[0]] + jet_ps_nob[pair[1]]).M() for pair in pairs}
						mjjs = dict(sorted(mjjs.items(), key=lambda item: item[1], reverse=True))
						sel_pair, mjj = next(iter(mjjs.items()))
						deta = jet_ps_nob[next(iter(mjjs))[0]].Eta() - jet_ps_nob[next(iter(mjjs))[1]].Eta()
					elif args.dnn:
						if args.m_id == 1:
							m_comb = re.search(r"MX_\d+_MY_\d+", sample)
							m_comb = m_comb.group(0) if m_comb else "MX_1000_MY_125"
							model = get_model(model_dict[m_comb], 1)
						else:
							if len(model_list)==1:
								model = get_model(model_list[0], args.m_id)
							else:
								print('Error: expected to find only one DNN model\n instead', model_list)
								sys.exit(1)
						model.eval()
						pairs_vars = {}
						for _, row in df_res.replace([np.inf, -np.inf], np.nan).iterrows():
							used_vars = open(dataset_vars_dict[m_comb]).read().splitlines()
							with open(norm_stats_dict[m_comb], 'r') as f:
								stats = json.load(f)
							with open(dataset_logvars_dict[m_comb], 'r') as f:
								log_vars = json.load(f)
							for var, cut in log_vars.items():
								row[var] = np.log(row[var] - cut + 1)
							abs_vars = open(dataset_absvars_dict[m_comb]).read().splitlines()

							for var in abs_vars:
								row[var] = np.abs(row[var])

							assert list(used_vars) == list(stats['mean'].keys()) == list(stats['std'].keys()), "variable order mismatch"
							norm_vars = (row[used_vars].values - np.asarray(list(stats['mean'].values()))) / np.asarray(list(stats['std'].values())) # trainval
							
							pairs_vars[int(row['id1']), int(row['id2'])] = np.nan_to_num(norm_vars, nan=0)
						pair_scores = {p: model(torch.tensor(v, dtype=torch.float32).unsqueeze(0)).detach().numpy().item() for p, v in pairs_vars.items()}

						for v in pair_scores.values():
							if np.isnan(v):
								print('error! nan score')
								sys.exit(1)

						pair_scores = dict(sorted(pair_scores.items(), key=lambda item: item[1], reverse=True))

						sel_pair, sel_pair_score = next(iter(pair_scores.items()))
						if passed and len(vbf_part_ids) > 1:
							for pair, score in pair_scores.items():
								if pair == vbf_pair:
									sig_count.append(score)
								else:
									bkg_count.append(score)
						mjj = (jet_ps_nob[sel_pair[0]]+jet_ps_nob[sel_pair[1]]).M()
						deta = jet_ps_nob[sel_pair[0]].Eta() - jet_ps_nob[sel_pair[1]].Eta()
						

						mjjs = {pair: (jet_ps_nob[pair[0]] + jet_ps_nob[pair[1]]).M() for pair in pairs}
						mjjs = dict(sorted(mjjs.items(), key=lambda item: item[1], reverse=True))
						mjj_sel_pair, _ = next(iter(mjjs.items()))
						mjj_dnn_match.append(True if (sel_pair == mjj_sel_pair) else False)


					# VBF cut
					if (mjj > 400) and (abs(deta) > 3) and passed:
						vbf_cut_passed +=1
					else:
						passed_vbf_cut = False

					# Save dnn eff
					if args.dnn:
						dnn_eff.append([unq_evt_id, sel_pair, sel_pair_score, passed, vbf_cut_passed])

					# Save bjets
					bjets_rows.append([
						unq_evt_id, passed, vbf_cut_passed,
						b_jets[0].Pt(), b_jets[0].Eta(), b_jets[0].Phi(), b_jets[0].M(), bJetsScores[0],
						b_jets[1].Pt(), b_jets[1].Eta(), b_jets[1].Phi(), b_jets[1].M(), bJetsScores[1],
						b_jets[2].Pt(), b_jets[2].Eta(), b_jets[2].Phi(), b_jets[2].M(), bJetsScores[2],
						b_jets[3].Pt(), b_jets[3].Eta(), b_jets[3].Phi(), b_jets[3].M(), bJetsScores[3],
						p_X.Pt(), p_X.Eta(), p_X.Phi(), p_X.M()
					])

					mjjs_blwp.append(mjj)
					detas_blwp.append(deta)
					if passed_vbf_cut:
						mjjs_vbfCut.append(mjj)
						detas_vbfCut.append(deta)

					pres = [Ht, mjj, deta]

					trg_masks = {name: getattr(event, name) for name in tnames}

					for trg_name, tname in zip(trigger_names, tnames):
							for k, i in zip(trg_dicts[trg_name].keys(), range(len(pres))):
									trg_dicts[trg_name][k] = ak.concatenate([trg_dicts[trg_name][k], ak.Array([pres[i] if trg_masks[tname] else np.nan])])

					Hts_pre = ak.concatenate([Hts_pre, [Ht]])
					mjjs_pre = ak.concatenate([mjjs_pre, [mjj]])
					detas_pre = ak.concatenate([detas_pre, [deta]])


					if flab == 'VBF' and passed and len(vbf_part_ids) > 1:

						purity_denominator += 1

						if sel_pair == vbf_pair:
							DeltaR_1, DeltaR_2 = vbf_dists.values()
							purity_test += 1
							mjjs_blwp_pure.append(mjj)
							detas_blwp_pure.append(deta)
							if passed_vbf_cut:
								mjjs_vbfCut_pure.append(mjj)
								detas_vbfCut_pure.append(deta)
						else:
							jet_alpha, jet_beta = jet_ps_nob[sel_pair[0]], jet_ps_nob[sel_pair[1]]
							DeltaR_AC, DeltaR_AD = jet_alpha.DeltaR(part_gamma, True), jet_alpha.DeltaR(part_delta, True)
							DeltaR_BC, DeltaR_BD = jet_beta.DeltaR(part_gamma, True), jet_beta.DeltaR(part_delta, True)
							if DeltaR_AC < DeltaR_AD:
								DeltaR_1 = DeltaR_AC
								c_flag = True
							else: 
								DeltaR_1 = DeltaR_AD
								c_flag = False
							if DeltaR_BC < DeltaR_BD:
								DeltaR_2 = DeltaR_BC
							else: 
								DeltaR_2 = DeltaR_BD

						jet_blwp_part_DeltaR_1_s.append(DeltaR_1)
						jet_blwp_part_DeltaR_2_s.append(DeltaR_2)
						if vbf_cut_passed:
							jet_vbfCut_part_DeltaR_1_s.append(DeltaR_1)
							jet_vbfCut_part_DeltaR_2_s.append(DeltaR_2)
						
						jet_blwp_part_flav_1_s.append(jet_nob_partFlavs[sel_pair[0]])
						jet_blwp_part_flav_2_s.append(jet_nob_partFlavs[sel_pair[1]])
						if vbf_cut_passed:
							jet_vbfCut_part_flav_1_s.append(jet_nob_partFlavs[sel_pair[0]])
							jet_vbfCut_part_flav_2_s.append(jet_nob_partFlavs[sel_pair[1]])

						if args.dnn:
							pur_rows.append([sel_pair == vbf_pair, sel_pair_score, DeltaR_1, DeltaR_2])


	# Bookkeeper: map events to root files
	bookeeping_dir = root_dir+("debug/" if args.debug else "")+'vbf_evt_ids/'
	os.makedirs(bookeeping_dir, exist_ok=True)
	with open(bookeeping_dir+f"paths_evt_ids_{sample}.json", "w") as f:
		json.dump(paths_evt_ids, f, indent=4)


	# Saving
	if not os.path.exists(pre_npz_name) or args.ow:
		if verbose:
			print('saving pres: ', pre_npz_name)
		np.savez(pre_npz_name, ht=Hts_pre, mjj=mjjs_pre, deta=detas_pre)
	for trg_name in trigger_names:
		if not os.path.exists(trg_npz_names[trg_name]) or args.ow:
			if verbose:
				print('saving posts: ', trg_npz_names[trg_name])
			np.savez(trg_npz_names[trg_name], ht=trg_dicts[trg_name]['ht'], mjj=trg_dicts[trg_name]['mjj'], deta=trg_dicts[trg_name]['deta'])
	
	if not os.path.exists(blwp_npz_name) or args.ow:
		if verbose:
			print('saving blwp evts: ', blwp_npz_name)
		np.savez(blwp_npz_name, mjj=mjjs_blwp, deta=detas_blwp)
	if not os.path.exists(blwp_pure_npz_name) or args.ow:
		if verbose:
			print('saving blwp pure evts: ', blwp_pure_npz_name)
		np.savez(blwp_pure_npz_name, mjj=mjjs_blwp_pure, deta=detas_blwp_pure)

	if not os.path.exists(vbfCut_npz_name) or args.ow:
		if verbose:
			print('saving vbf evts: ', vbfCut_npz_name)
		np.savez(vbfCut_npz_name, mjj=mjjs_vbfCut, deta=detas_vbfCut)
	if not os.path.exists(vbfCut_pure_npz_name) or args.ow:
		if verbose:
			print('saving vbf pure evts: ', vbfCut_pure_npz_name)
		np.savez(vbfCut_pure_npz_name, mjj=mjjs_vbfCut_pure, deta=detas_vbfCut_pure)

	if has_VBF:
		if not os.path.exists(jetMatches_vbfCut_npz_fname) or args.ow:
			np.savez(jetMatches_vbfCut_npz_fname, DeltaR_1 = jet_vbfCut_part_DeltaR_1_s, DeltaR_2 = jet_vbfCut_part_DeltaR_2_s, 
											Jet1_flav = jet_vbfCut_part_flav_1_s, Jet2_flav = jet_vbfCut_part_flav_2_s,)
		if not os.path.exists(jetMatches_blwp_npz_fname) or args.ow:
			np.savez(jetMatches_blwp_npz_fname, DeltaR_1 = jet_blwp_part_DeltaR_1_s, DeltaR_2 = jet_blwp_part_DeltaR_2_s, 
											Jet1_flav = jet_blwp_part_flav_1_s, Jet2_flav = jet_blwp_part_flav_2_s,)
		if not os.path.exists(parquet_fname) or args.ow:
			df.to_parquet(parquet_fname, index=False)
		if args.dnn:
			if not os.path.exists(purpar_fname) or args.ow:
				datafr = pd.DataFrame(pur_rows, columns=['is_vbf', 'selpair_dnnscore', 'DeltaR1', 'DeltaR2'])
				datafr.to_parquet(purpar_fname, index=False)
	if args.dnn:
		if not os.path.exists(dnn_eff_fname) or args.ow:
			dataf = pd.DataFrame(dnn_eff, columns=['evt', 'pair', 'score', 'passed', 'vbf_cut_passed'])
			dataf.to_parquet(dnn_eff_fname, index=False)

	if not os.path.exists(bjets_fname) or args.ow:
		dtfr = pd.DataFrame(bjets_rows, columns=[
			'evt', 'passed', 'vbf_cut_passed', 
			'pt_0', 'eta_0', 'phi_0', 'm_0', 'score_0',
			'pt_1', 'eta_1', 'phi_1', 'm_1', 'score_1',
			'pt_2', 'eta_2', 'phi_2', 'm_2', 'score_2',
			'pt_3', 'eta_3', 'phi_3', 'm_3', 'score_3',
			'pt_X', 'eta_X', 'phi_X', 'm_X'
		])
		dtfr.to_parquet(bjets_fname, index=False)



	writer.writerow([sample, total_events, total_passed_any, jet_num_cut_passed, looseWP_cut_passed, vbf_cut_passed, purity_test, purity_denominator])
	print()


if args.debug and args.dnn:
	from matplotlib import pyplot as plt
	plt.hist(sig_count, histtype='step', bins=np.linspace(0., 1.+0.025, 40), density=True, label='sig')
	plt.hist(bkg_count, histtype='step', bins=np.linspace(0., 1.+0.025, 40), density=True, label='bkg')
	plt.legend()
	plt.savefig('temp_hist.png')

	print('mjj-dnn match', sum(mjj_dnn_match)/len(mjj_dnn_match))
	print('vbf present at GenPart level', sum(is_there_a_vbf_pair)/len(is_there_a_vbf_pair))

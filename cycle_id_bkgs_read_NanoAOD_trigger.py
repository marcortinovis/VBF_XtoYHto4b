import uproot
import awkward as ak
import glob
import argparse
from tqdm import tqdm
import csv
from paths import *

bkgs = ['QCD']
bkgs_paths = {'QCD': QCD_PATH}

print(bkgs_paths)


trigger_names = [
	# Example triggers
	"HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepJet_4p5", # Analogous to Run 2	# 000
	"HLT_PFHT250_QuadPFJet25_PNet2BTagMean0p55", # Targeting ggF HH							# 001
	# VBF HH
	# Standard stream
	"HLT_VBF_DiPFJet125_45_Mjj1050",														# 002
	"HLT_VBF_DiPFJet75_45_Mjj800_DiPFJet60",												# 003
	# Parking stream
	"HLT_QuadPFJet103_88_75_15",															# 004
	"HLT_QuadPFJet103_88_75_15_PNet2BTag_0p4_0p12_VBF1",									# 005
	"HLT_QuadPFJet103_88_75_15_PNetBTag_0p4_VBF2",											# 006
]

parser = argparse.ArgumentParser(description="Read NanoAOD files and calculate trigger efficiencies.")
parser.add_argument("--sample", type=str, help="Bkg sample to compute the efficiency on.")
parser.add_argument("--all", action="store_true", help="Performs a full scan using all the bkgs available.")
parser.add_argument("--i", type=int, help="Id of the first trigger.", required=False)
parser.add_argument("--d", type=int, help="Id of the second trigger.", required=False)
parser.add_argument("--label", type=str, help="Custom label for filename")
parser.add_argument("--ow", action="store_true", help="Overwrite already existent files.")
parser.add_argument("--debug", action="store_true", help="Debug: runs script on just a few events.")
args = parser.parse_args()

if args.i is None:
	trigger_names = trigger_names
else:
	trigger_names = [trigger_names[args.i]] + ([trigger_names[args.d]] if args.d is not None else [])

if (args.sample is None) and not args.all:
	print("You should specify MX and MY if the --all flag is not present!")
	raise SystemExit
if args.all:
	samples = bkgs	
	csv_filename = f"csvs/bkg_scan_"
	if args.i is not None:
		csv_filename += f"tr_{args.i:03d}" + (f"_{args.d:03d}" if args.d is not None else "" )
	elif args.label is not None:
		csv_filename += args.label
	csv_filename += ".csv"
	csv_fieldnames = ["sample", "tot_n_evts", *trigger_names, *['excl_'+t for t in trigger_names], "any", "all"]
	try:
		file = open(csv_filename, mode="w" if args.ow else "x", newline="")
		writer = csv.writer(file)
		writer.writerow(csv_fieldnames)
	except FileExistsError :
		if not args.ow:
			print(f"Error: {csv_filename} already exists. Exiting program.")
			raise SystemExit
elif args.sample is not None:
	samples = [args.sample]

for sample in samples:
	print(f"Running for sample {sample}.")

	file_pattern = bkgs_paths[sample]
	file_paths = glob.glob(file_pattern)

	print(f"Found {len(file_paths)} files.")

	total_events = 0
	individual_trigger_counts = {name: 0 for name in trigger_names}
	exclusive_trigger_counts = {name: 0 for name in trigger_names}
	total_passed_any = 0
	total_passed_all = 0

	if args.debug:
		file_paths = file_paths[:2] ## just for testing
	for path in tqdm(file_paths, desc="Processing NanoAODs", unit="file"):
		with uproot.open(f"{path}:Events") as tree:
			batch = tree.arrays(['hlt'+t[3:] for t in trigger_names], library="ak")
			batch = ak.zip({'HLT'+f[3:]: batch[f] for f in batch.fields})
			total_events += len(batch)
			n_trgs_passed = sum(batch[name] for name in trigger_names)

			for name in trigger_names:
				individual_trigger_counts[name] += ak.sum(batch[name])
				exclusive_trigger_counts[name] += ak.sum(batch[name] & (n_trgs_passed == 1))

			any_passed_mask = batch[trigger_names[0]]
			all_passed_mask = batch[trigger_names[0]]

			for name in trigger_names[1:]:
				any_passed_mask = any_passed_mask | batch[name]
				all_passed_mask = all_passed_mask & batch[name]

			total_passed_any += ak.sum(any_passed_mask)
			total_passed_all += ak.sum(all_passed_mask)

	print("\n" + "="*50)
	print("--- Trigger Efficiency Summary ---")
	print("="*50)
	print(f"Total events processed: {total_events:,}")
	print("-" * 50)

	print("Individual Counts (Double-counting allowed):")
	for name, count in individual_trigger_counts.items():
		efficiency = (count / total_events) * 100 if total_events > 0 else print("N/A")
		print(f"  - {name}: {count:,} (Efficiency: {efficiency:.2f}%)")

	print("-" * 50)

	print("Exclusive Individual Counts:")
	for name, count in exclusive_trigger_counts.items():
		efficiency = (count / total_events) * 100 if total_events > 0 else print("N/A")
		print(f"  - {name}: {count:,} (Efficiency: {efficiency:.2f}%)")

	print("-" * 50)

	print(f"Unique events triggered (Pass ANY): {total_passed_any:,}")
	if total_events > 0:
		efficiency_any = (total_passed_any / total_events) * 100
		print(f"Global Trigger Efficiency (OR): {efficiency_any:.2f}%")
	print("="*50)

	print("-" * 50)
	print(f"Unique events triggered (Pass ALL): {total_passed_all:,}")
	if total_events > 0:
		efficiency_all = (total_passed_all / total_events) * 100
		print(f"Global Trigger Efficiency (AND): {efficiency_all:.2f}%")
	print("="*50)


	writer.writerow([sample, total_events, *list(individual_trigger_counts.values()), *list(exclusive_trigger_counts.values()), total_passed_any, total_passed_all])

file.close()




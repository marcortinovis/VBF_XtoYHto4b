import shap
import os
import torch
from pathlib import Path
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from models.dnn import *
from plot_metrics import *
import argparse

DNN_FUNCTIONS = {
	int(name.split("_")[1]): func
	for name, func in globals().items()
	if name.startswith("dnn_") and callable(func)
}

def get_model(model_path, id):
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

class ShapModel(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model
	def forward(self, x):
		return self.model(x).unsqueeze(1)



parser = argparse.ArgumentParser(description="Train the DNN for VBF Jet identification.")
parser.add_argument("--MX", type=str, default=None, help="Mass of the X particle (MX).")
parser.add_argument("--MY", type=str, default=None, help="Mass of the Y particle (MY).")
parser.add_argument("--Mall",  action="store_true", default=None, help="Run the training for all the mass combinations.")
parser.add_argument('--m_id', type=int, required=True, help="Choose the model type")
parser.add_argument('--d_id', type=str, required=True, help="Choose the dataset type")
#parser.add_argument("--debug", action="store_true", help="Debug: runs script on just a few events.")
args = parser.parse_args()

if args.Mall is None:
	if args.MX is None or args.MY is None:
		parser.error("Either --Mall or both --MX and --MY must be provided")
elif args.MX is not None or args.MY is not None:
	parser.error("--Mall cannot be used with --MX or --MY")

m_combs = ['MX_1000_MY_125',
		 'MX_500_MY_125', 'MX_700_MY_125', 'MX_1400_MY_125', 'MX_2000_MY_125',
		 'MX_1000_MY_90', 'MX_1000_MY_200', 'MX_1000_MY_400', 'MX_1000_MY_600']
if args.Mall is None:
	m_combs = [m_comb for m_comb in m_combs if (args.MX in m_comb and args.MY in m_comb)]

cols_to_drop = ['MX', 'MY', 'event', 'id1', 'id2']

model_list = [
	file
	for directory in Path('results/').rglob("*")
	if directory.is_dir() and f"{args.m_id:03d}" in directory.name
	for file in directory.glob("*.pt")
]
if args.m_id == 1:
	model_dict = {
		m.group(): filename
		for filename in model_list
		if (m := re.search(r"MX_\d+_MY_\d+", str(filename)))
	}


for m_comb in m_combs:

	# Load

	out_dir = f'results/results_{args.m_id:03d}/results_{args.m_id:03d}_{m_comb}/shap_explanation/'
	os.makedirs(out_dir, exist_ok=True)

	train_set = EventDataset(f'datasets/datasets_{args.d_id}/dataset_{args.d_id}_{m_comb}/{m_comb}_train_set.parquet', cols_to_drop=cols_to_drop)
	train_dataloader = DataLoader(train_set, batch_size=1024, shuffle=False)
	print('train set loaded')

	test_set = EventDataset(f'datasets/datasets_{args.d_id}/dataset_{args.d_id}_{m_comb}/{m_comb}_test_set.parquet', cols_to_drop=cols_to_drop)
	test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
	print('test set loaded')

	y_train, X_train = next(iter(train_dataloader))
	y_test, X_test = next(iter(test_dataloader))

	print('X_train', len(X_train), 'X_test', len(X_test))

	model =	get_model(model_dict[m_comb], args.m_id)
	print('model loaded')


	# Explaining

	shap_model = ShapModel(model).eval()

	explainer = shap.DeepExplainer(shap_model, X_train)
	print('explainer created')

	shap_batch_size = 32
	shap_batches = []
	for i in tqdm(range(0, len(X_test), shap_batch_size), desc=f"SHAP {m_comb}"):
		X_batch = X_test[i:i + shap_batch_size]
		values = explainer.shap_values(X_batch)
		if isinstance(values, list):
			values = values[0]
		if torch.is_tensor(values):
			values = values.detach().cpu().numpy()
		if values.ndim == 3 and values.shape[-1] == 1:
			values = values[:, :, 0]
		shap_batches.append(values)
	shap_values = np.concatenate(shap_batches, axis=0)
	print('explanation finished')


	# Saving

	shap_data = {
		"shap_values": shap_values,
		"data": X_test.detach().cpu().numpy(),
		"y_test": y_test.detach().cpu().numpy(),
	}
	shap_feats = train_set.labs[1:]

	np.savez_compressed(
		os.path.join(out_dir, f"{args.m_id:03d}_{m_comb}_shap_values.npz"),
		**shap_data
	)
	print("SHAP data saved")
	with open(out_dir+f"{args.m_id:03d}_{m_comb}_shap_features.txt", "w") as f:
		f.write("\n".join(shap_feats))
	print("SHAP feats saved")

	plot_shap(f'{args.m_id:03d}')	

	print(f'done with {m_comb}')
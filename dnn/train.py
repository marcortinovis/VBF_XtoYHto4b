# --- Imports ---

from models.dnn import *
from plot_metrics import *
from tqdm import tqdm
import os
import pandas as pd
import argparse
from datetime import datetime

# --- Params ---

parser = argparse.ArgumentParser(description="Train the DNN for VBF Jet identification.")
parser.add_argument("--m_id", type=int, required=True)
parser.add_argument("--MX", type=str, default=None, help="Mass of the X particle (MX).")
parser.add_argument("--MY", type=str, default=None, help="Mass of the Y particle (MY).")
parser.add_argument("--Mall",  action="store_true", default=None, help="Run the training for all the mass combinations.")
parser.add_argument("--layers", nargs="+", type=int, required=True)
parser.add_argument("-bs", "--batch_size", type=int, default=1024)
parser.add_argument("-lr", "--learning_rate", type=float, default=1.e-3)
parser.add_argument("-ne", "--number_of_epochs", type=int, default=100)
parser.add_argument("-esp", "--EarlyStopping_patience", type=int, default=5)
parser.add_argument("-esd", "--EarlyStopping_delta", type=float, default=1.e-4)
parser.add_argument("-esm", "--EarlyStopping_metric", type=str, default='validation_loss')
parser.add_argument("--debug", action="store_true", help="Debug: runs script on just a few events.")
args = parser.parse_args()

if args.m_id == 1 or args.m_id == 4:
	d_id = 'alpha'
elif args.m_id == 2 or args.m_id == 3:
	d_id = 'beta'

if args.m_id==1 or args.m_id == 4:
	if args.Mall is None:
		if args.MX is None or args.MY is None:
			parser.error("Either --Mall or both --MX and --MY must be provided")
	elif args.MX is not None or args.MY is not None:
		parser.error("--Mall cannot be used with --MX or --MY")

m_combs = ['MX_1000_MY_125',
		 'MX_500_MY_125', 'MX_700_MY_125', 'MX_1400_MY_125', 'MX_2000_MY_125',
		 'MX_1000_MY_90', 'MX_1000_MY_200', 'MX_1000_MY_400', 'MX_1000_MY_600']
if args.m_id==2 or args.m_id==3:
	m_combs = ['Mall']
elif args.Mall is None:
	m_combs = [m_comb for m_comb in m_combs if (args.MX in m_comb and args.MY in m_comb)]

cols_to_drop = ['event', 'id1', 'id2']
if args.m_id != 3:
	cols_to_drop += ['MX', 'MY']
batch_size = args.batch_size

# --- Definitions ---

class EarlyStopping:
	def __init__(self, patience=5, min_delta=1e-4, monitor='validation_loss'):
		self.patience = patience
		self.min_delta = min_delta
		self.monitor = monitor
		self.best = float('inf')
		self.counter = 0
		self.best_epoch = 0
		self.best_state = None  # stores best weights in memory

	def step(self, metrics, model):
		score = metrics[self.monitor]
		if score < self.best - self.min_delta:
			self.best = score
			self.counter = 0
			self.best_epoch = metrics['epoch']
			self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
			return False 
		else:
			self.counter += 1
			return self.counter >= self.patience

	def restore_best(self, model):
		if self.best_state:
			model.load_state_dict(self.best_state)
			print(f"Restored best weights from epoch {self.best_epoch} (val_loss={self.best:.4f})")

def training_step(model, trn_drldr, vld_dtldr, opt):

	trn_correct = 0.
	trn_total = 0.
	trn_loss = 0.
	model.train()

	for i, trn_data in enumerate(trn_drldr):

		y_trn, X_trn = trn_data
		opt.zero_grad()
		trn_out = model(X_trn)

		trn_pred = (trn_out >= 0.5).float()
		trn_correct += (trn_pred == y_trn).sum().item()
		trn_total += y_trn.numel()

		trn_loss_batch = criterion(trn_out, y_trn)
		trn_loss_batch.backward()
		trn_loss += trn_loss_batch.item()
		opt.step()

	trn_loss = trn_loss / len(trn_drldr)
	trn_acc = trn_correct / trn_total


	vld_correct = 0.
	vld_total = 0.
	vld_loss = 0.
	model.eval()

	with torch.no_grad():
		for j, vld_data in enumerate(vld_dtldr):

			y_vld, X_vld = vld_data
			vld_out = model(X_vld)

			vld_pred = (vld_out >= 0.5).float()
			vld_correct += (vld_pred == y_vld).sum().item()
			vld_total += y_vld.numel()

			vld_loss += criterion(vld_out, y_vld).item()

	vld_loss = vld_loss / len(vld_dtldr)
	vld_acc = vld_correct / vld_total

	return {'training_loss': trn_loss, 'validation_loss': vld_loss,
		'training_accuracy': trn_acc, 'validation_accuracy': vld_acc}


for m_comb in m_combs:

	# --- Loading ---

	train_set = EventDataset(f'datasets/datasets_{d_id}/dataset_{d_id}_{m_comb}/{m_comb}_train_set.parquet', cols_to_drop=cols_to_drop)
	train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

	val_set = EventDataset(f'datasets/datasets_{d_id}/dataset_{d_id}_{m_comb}/{m_comb}_val_set.parquet', cols_to_drop=cols_to_drop)
	val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

	test_set = EventDataset(f'datasets/datasets_{d_id}/dataset_{d_id}_{m_comb}/{m_comb}_test_set.parquet', cols_to_drop=cols_to_drop)
	test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


	y, X = next(iter(train_dataloader))


	# --- Creation ---

	DNN_FUNCTIONS = {
		int(name.split("_")[1]): func
		for name, func in globals().items()
		if name.startswith("dnn_") and callable(func)
	}

	dnn = DNN_FUNCTIONS[args.m_id](X.shape[-1], args.layers)
	print(dnn)
	print(X[0])

	optimizer = torch.optim.Adam(dnn.parameters(), lr=args.learning_rate)
	criterion = nn.BCELoss()
	early_stopper = EarlyStopping(patience=args.EarlyStopping_patience, min_delta=args.EarlyStopping_delta)


	# --- Training ---
	print('Starting')
	print(f'Training set size {len(train_set)}, Validation set size {len(val_set)}')
	history = []
	pbar = tqdm(range(args.number_of_epochs), desc="Training")
	for epoch in pbar:
		h = training_step(dnn, train_dataloader, val_dataloader, optimizer)
		h = {'epoch': epoch, **h}
		history.append(h)
		tqdm.write(
			f"Epoch {epoch:03d} | "
			f"Train loss {h['training_loss']:6.3f} | "
			f"Val loss {h['validation_loss']:6.3f} | "
			f"Train acc {h['training_accuracy']:6.3f} | "
			f"Val acc {h['validation_accuracy']:6.3f}"
		)
		if early_stopper.step(h, dnn):
			tqdm.write(f"Early stopping at epoch {epoch} (no improvement for {early_stopper.patience} epochs)")
			break 

	early_stopper.restore_best(dnn)


	# --- Results ---

	out_dir = ('debug/' if args.debug else '')+f'results/results_{args.m_id:03d}/results_{args.m_id:03d}_{m_comb}/'
	os.makedirs(out_dir, exist_ok=True)

	# Log
	# Save calculation parameters
	info_file = out_dir + "info.txt"

	with open(info_file, "w", encoding="utf-8") as f:
		f.write("Training parameters information\n")
		f.write("=======================\n\n")
		f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
		f.write(f"Script: {Path(__file__).name}\n\n")

		f.write("Parameters:\n")
		for name, value in vars(args).items():
			f.write(f"  {name} = {value}\n")

	# Model
	torch.save(dnn.state_dict(), out_dir+f'model_{args.m_id:03d}_{m_comb}.pt')

	# History
	df = pd.DataFrame(history)
	df.to_csv(out_dir+f'history_{args.m_id:03d}_{m_comb}.csv', index=False)

	# Outputs
	train_outs, val_outs, test_outs = [], [], []
	dnn.eval()
	with torch.no_grad():
		for y_trn, X_trn in train_dataloader:
			trn_out = dnn(X_trn)
			train_outs.append(torch.stack((y_trn, trn_out), dim=1).cpu())
		for y_vld, X_vld in val_dataloader:
			vld_out = dnn(X_vld)
			val_outs.append(torch.stack((y_vld, vld_out), dim=1).cpu())
		for y_tst, X_tst in test_dataloader:
			tst_out = dnn(X_tst)
			test_outs.append(torch.stack((y_tst, tst_out), dim=1).cpu())
	train_outs = torch.cat(train_outs, dim=0).numpy()
	npz_train_outs = out_dir+f"train_outs_{args.m_id:03d}_{m_comb}.npz"
	np.savez(npz_train_outs, y_true=train_outs[:, 0], y_pred=train_outs[:, 1])
	print(f"{npz_train_outs} saved")
	val_outs = torch.cat(val_outs, dim=0).numpy()
	npz_val_outs = out_dir+f"val_outs_{args.m_id:03d}_{m_comb}.npz"
	np.savez(npz_val_outs, y_true=val_outs[:, 0], y_pred=val_outs[:, 1])
	print(f"{npz_val_outs} saved")
	test_outs = torch.cat(test_outs, dim=0).numpy()
	npz_test_outs = out_dir+f"test_outs_{args.m_id:03d}_{m_comb}.npz"
	np.savez(npz_test_outs, y_true=test_outs[:, 0],	y_pred=test_outs[:, 1])
	print(f"{npz_test_outs} saved")

	plot_all(f'{args.m_id:03d}')
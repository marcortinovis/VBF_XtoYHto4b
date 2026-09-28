import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd




class EventDataset(Dataset):    
    # Dataset class, only to load it

    def __init__(self, filename, length=None, cols_to_drop=None):
        data = pd.read_parquet(filename)
        if cols_to_drop is not None:
            data.drop(columns=cols_to_drop, inplace=True)
        if length is not None:
            if length > len(data):
                raise ValueError(
                    f"length ({length}) cannot be greater than "
                    f"the number of samples ({len(data)})"
                )
            data = data.sample(n=length)
        self.labs = data.columns
        self.y = torch.from_numpy(data['is_vbf'].to_numpy()).float()
        data.drop(columns=['is_vbf'], inplace=True)
        self.X = torch.from_numpy(data.to_numpy()).float()
        self.X.nan_to_num_(nan=0.0)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.y[idx], self.X[idx]



class dnn_001(nn.Module):
    # First architecture

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        dims = [input_dim]+hidden_dims+[1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(1)

    
class dnn_02(dnn_001):
    # Same architecture
    pass


class dnn_03(dnn_001):
    # Same architecture
    pass


class dnn_004(nn.Module):
    # No dropout

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        dims = [input_dim]+hidden_dims+[1]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                #layers.append(nn.Dropout(0.2))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(1)
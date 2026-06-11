import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource, get_stocks
import joblib
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import gc
import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from models import Time2Vec
from tqdm import tqdm
import signal

from experiments import Experiment

#stocks = get_stocks()
#print("Finalizing datasets...")
#for stock in stocks:
#    stock_data = DataSource()
#    stock_data.create_df(file_name=stock, medium='final', target=30, ignore_history=True)
#    del stock_data
#    gc.collect()
#
#for stock in stocks:
#    stock_data = DataSource()
#    stock_data.create_df(file_name=stock, medium='final', target=10, ignore_history=True)
#    del stock_data
#    gc.collect()

class Time2VecDataset(Dataset):
    def __init__(self, path, stock_lookback):
        self.stock_data = torch.load(path, map_location=torch.device('cpu'))
        self.stock_lookback = stock_lookback
    
    def __len__(self):
        return self.stock_data['features'].shape[1] - self.stock_lookback + 1
    
    def __getitem__(self, idx):
        x = self.stock_data['features'][:, idx:idx + self.stock_lookback, :]
        y = self.stock_data['target'][:, :, idx]
        t = self.stock_data['timestamps'][:, idx:idx + self.stock_lookback]
        m = self.stock_data['mask'][:, idx:idx + self.stock_lookback]

        # print(f'Shapes: X: {x.shape}; y: {y.shape}; ts: {t.shape}; m: {m.shape}')

        return t, y

class Time2VecModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.time2vec = Time2Vec(input_dim)
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, t):
        t = self.time2vec(t) # (B, S, 60) -> (B, S, 60, 32)
        return self.linear(t[:, :, -1, :]) # (B, S, 60, 32) -> (B, S, 32) -> (B, S, 2)

    
def collate_fn(batch):
    args = list(zip(*batch))
    n = len(args)

    for i, arg in enumerate(args[:n]):
        lengths = torch.tensor([len(f) for f in arg])

        if not (lengths == lengths[0]).all():
            args[i] = pad_sequence(arg, batch_first=True, padding_value=0.0)

            _, L_max = args[i].shape[:2]
            arg_mask = torch.arange(L_max).unsqueeze(0) < lengths.unsqueeze(1)

            args.insert(-1, arg_mask)
        
        else:
            args[i] = torch.stack(list(arg))

    return tuple(args)

def _run_validation(model, loaders, device, criterion):
    model.eval()
    total_val_loss = 0

    with tqdm(total=len(loaders['val']), desc="Validation") as pbar:

        with torch.no_grad():
            for *args, target in loaders['val']:
                target = target.argmax(dim=-1).to(device)

                args = [a.to(device) for a in args]

                logits = model(*args)[0]
                logits = logits.permute(0, 2, 1)

                loss = criterion(logits, target)
                total_val_loss += loss.item()

                pbar.update(1)
    
    model.train()

    return total_val_loss / len(loaders['val'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Time2VecModel(32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

loaders = {
    split: DataLoader(
        Time2VecDataset('experiments/data/stock_transformer_30m_train.pt', 60),
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    for split in ('train', 'val', 'test')
}

for *args, target in loaders['train']:
    args = [a.to(device) for a in args]
    t = args[0]
    
    print("t stats:", t.min().item(), t.max().item(), t.mean().item(), torch.isnan(t).any().item())
    
    out = model.time2vec(t)
    print("after time2vec:", out.min().item(), out.max().item(), torch.isnan(out).any().item())
    
    out2 = model.linear(out[:, :, -1, :])
    print("after linear:", out2.min().item(), out2.max().item(), torch.isnan(out2).any().item())
    
    loss = criterion(out2.permute(0, 2, 1), target.argmax(dim=-1).to(device))
    print("loss:", loss.item())
    
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"grad {name}:", param.grad.min().item(), param.grad.max().item(), torch.isnan(param.grad).any().item())
    
    optimizer.step()
    
    # check params after update
    for name, param in model.named_parameters():
        print(f"param {name} after update:", param.min().item(), param.max().item(), torch.isnan(param).any().item())
    

for *args, target in loaders['train']:
    args = [a.to(device) for a in args]
    t = args[0]
    
    out = model.time2vec(t)
    print("second batch after time2vec:", torch.isnan(out).any().item())
    break

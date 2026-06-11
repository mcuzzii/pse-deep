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

def train(
    model,
    num_epochs=3,
    batch_size=32,
    lr=1e-3,
    val_every=500
):
    path = Path('experiments/stock_transformer/stock_transformer_30.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders = {
        split: DataLoader(
            Time2VecDataset('experiments/data/stock_transformer_30m_train.pt', 60),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn
        )
        for split in ('train', 'val', 'test')
    }

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    total_loss = 0
    train_losses = []
    val_losses = []

    resume_step = None
    if path.exists():
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        resume_step = checkpoint["global_step"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        total_loss = checkpoint["total_loss"]

    pbar = tqdm(total=len(loaders['train']) * num_epochs, desc="Training")

    interrupted = False

    def handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("Interrupt received. Saving checkpoint...")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Training loop
    try:
        for epoch in range(num_epochs):
            model.train()

            for *args, target in loaders['train']:

                print(args[0].shape)
                print(target.shape)

                if interrupted:
                    raise KeyboardInterrupt
                
                if resume_step and global_step < resume_step:
                    global_step += 1
                    pbar.update(1)
                    continue

                elif resume_step and global_step >= resume_step:
                    resume_step = None  # done catching up

                target = target.argmax(dim=-1)       # (B, 30, 2) → (B, 30)

                target = target.to(device)
                args = [a.to(device) for a in args]

                optimizer.zero_grad()
                logits = model(*args)[0]       # (B, 30, 2)
                logits = logits.permute(0, 2, 1)     # (B, 2, 30)

                loss = criterion(logits, target)     # target (B, 30)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

                global_step += 1

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

                if global_step % val_every == 0:
                    val_loss = _run_validation(model, loaders, device, criterion)
                    val_losses.append(val_loss)

                    train_loss = total_loss / val_every
                    total_loss = 0
                    train_losses.append(train_loss)

                    pbar.set_postfix(train_loss=loss.item(), val_loss=val_loss)

                    # optional checkpoint on validation
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "total_loss": total_loss
                    }, path)
            
            print(f"Epoch {epoch + 1}/{num_epochs} ({len(loaders['train']) * (epoch + 1)} batches).")

    except KeyboardInterrupt:
        pass

    finally:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "total_loss": total_loss
        }, path)
        
        pbar.close()

model = Time2Vec(32)
train(model)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sys
import signal
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from models import StockTransformer
from processing import DataSource, get_stocks, get_text_window

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

class StockTransformerDataset(Dataset):
    def __init__(self, path, stock_lookback):
        self.stock_data = torch.load(path)
        self.stock_lookback = stock_lookback
    
    def __len__(self):
        return self.stock_data['features'].shape[1] - self.stock_lookback + 1
    
    def __getitem__(self, idx):
        x = self.stock_data['features'][:, idx:idx + self.stock_lookback, :]
        y = self.stock_data['target'][:, :, idx]
        t = self.stock_data['timestamps'][:, idx:idx + self.stock_lookback]
        m = self.stock_data['mask'][:, idx:idx + self.stock_lookback]

        # print(f'Shapes: X: {x.shape}; y: {y.shape}; ts: {t.shape}; m: {m.shape}')

        return t, x, m, y

class StockNewsTransformerDataset(StockTransformerDataset):
    def __init__(self, stock_path, news_path, stock_lookback, pred_horizon, time_vec_input):
        super().__init__(stock_path, stock_lookback)
        self.news_data = torch.load(news_path)

        self.pred_horizon = pred_horizon
        self.time_vec_input = time_vec_input
    
    def __len__(self):
        self.stock_data['features'].shape[0]
    
    def __getitem__(self, idx):
        t, x, m, y = super().__getitem__(idx)

        idx = (self.time_vec_input - t[-1]).abs().idxmin()
        cutoff, _ = get_text_window(idx, self.time_vec_input.index, self.pred_horizon)
        cutoff_scaled = self.time_vec_input[cutoff]

        embeddings = self.news_data['embeddings']
        timestamps = self.news_data['timestamps']
        window = (cutoff_scaled < timestamps) & (timestamps <= t[-1])

        # print(f'Shapes: news_e: {embeddings[window].shape}; news_t: {timestamps[window].shape}')

        return t, timestamps[window], x, embeddings[window], m, y

class Experiment:
    def __init__(
        self,
        experiment_name: str,
        transformer: bool = True,
        pred_30: bool = True,
        news: bool = True,
        social: bool = True,
        stock_lookback: int = 60
    ):
        self.transformer = transformer
        self.news = news
        self.social = social
        self.stock_lookback = stock_lookback
        self.pred_horizon = 30 if pred_30 else 10
        self.processed_path = Path('data/processed')
        self.experiment_path = Path(f'experiments/{experiment_name}')
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self.data_path = Path('experiments/data')
        self.data_path.mkdir(parents=True, exist_ok=True)

        reference_df = DataSource()
        reference_df.create_df(f'ac_{self.pred_horizon}m')

        self.filtered_date_times = reference_df.filtered_date_times
        self.train_cutoff = reference_df.train_cutoff
        self.val_cutoff = self.filtered_date_times[int(0.9 * len(self.filtered_date_times))]
        self.time_vec_input = reference_df.df[reference_df.time_vec_input]
    
    def _get_train_split(self, df):
        return df.loc[df.index.get_level_values('local_time') <= self.train_cutoff]

    def _get_val_split(self, df):
        last_train_idx = df.index.get_loc(self.train_cutoff)

        non_train_split = df.iloc[last_train_idx - self.stock_lookback + 2:]
        non_test_mask = non_train_split.index.get_level_values('local_time') <= self.val_cutoff

        return non_train_split.loc[non_test_mask]

    def _get_test_split(self, df):
        last_val_idx = df.index.get_loc(self.val_cutoff)

        return df.iloc[last_val_idx - self.stock_lookback + 2:]
    
    def build_model(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        embedding_dim: int | None = None,
        temporal_embedding_dim: int | None = None,
        num_heads: int = 8,
        num_layers: int = 1,
        expansion: int = 4,
        dropout: int | None = None
    ):
        if self.transformer and not self.news and not self.social:
            self.model = StockTransformer(
                input_dim,
                embedding_dim,
                temporal_embedding_dim,
                num_heads,
                num_layers,
                expansion,
                dropout
            )
    
    def _get_stock_shapes(self, num_sequences):
        X_shape = (num_sequences, len(self.stock_dfs), self.stock_lookback, len(self.stock_dfs[0].features))
        y_shape = (num_sequences, len(self.stock_dfs), 2)
        ts_shape = (num_sequences, len(self.stock_dfs), self.stock_lookback)
        m_shape = (num_sequences, len(self.stock_dfs), self.stock_lookback)

        return X_shape, y_shape, ts_shape, m_shape
    
    def _build_stock_transformer_data(self, force=False):

        train_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_train.pt'
        val_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_val.pt'
        test_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_test.pt'

        if train_path.exists() and val_path.exists() and test_path.exists() and not force:
            return
        
        stocks = get_stocks()

        self.stock_dfs = []
        for stock in stocks:
            print(f'Loading {stock}...')
            stock_df = DataSource()
            stock_df.create_df(file_name=f'{stock}_{self.pred_horizon}m')
            stock_df.df = stock_df.df.sort_index()
            self.stock_dfs.append(stock_df)

        for path, split_func in zip(
            [train_path, val_path, test_path],
            [self._get_train_split, self._get_val_split, self._get_test_split]
        ):

            tensors = {
                'timestamps': list(),
                'features': list(),
                'target': list(),
                'mask': list()
            }

            for stock_df in self.stock_dfs:

                split = split_func(stock_df.df)
                target = split[stock_df.target].iloc[self.stock_lookback - 1:].values

                stock_X = split[stock_df.features].values
                stock_y = np.array([target, 1 - target])
                stock_ts = split[stock_df.time_vec_input].values
                stock_m = split[stock_df.no_activity_col].values
                
                tensors['features'].append(stock_X)
                tensors['target'].append(stock_y)
                tensors['timestamps'].append(stock_ts)
                tensors['mask'].append(stock_m)

                print(
                    f'Processed {stock_df.file_name} into arrays:\n'
                    f'- Timestamps: {stock_ts.shape},\n'
                    f'- Features: {stock_X.shape},\n'
                    f'- Mask: {stock_m.shape},\n'
                    f'- Target: {stock_y.shape}.\n\n'
                )

            tensors['features'] = torch.from_numpy(np.array(tensors['features'], dtype=np.float32))         # (30, B, 100)
            tensors['target'] = torch.from_numpy(np.array(tensors['target'], dtype=np.float32))          # (30, 2, B)
            tensors['timestamps'] = torch.from_numpy(np.array(tensors['timestamps'], dtype=np.float32))     # (30, B)
            tensors['mask'] = torch.from_numpy(np.array(tensors['mask'], dtype=np.float32))                 # (30, B)

            print(
                'Final dataset sizes:\n'
                f'- Timestamps: {tensors['timestamps'].shape},\n'
                f'- Features: {tensors['features'].shape},\n'
                f'- Mask: {tensors['mask'].shape},\n'
                f'- Target: {tensors['target'].shape}.\n\n'
            )

            torch.save(tensors, path)
    
    def _build_news_transformer_data(self, force=False):

        train_path = self.data_path / f'news_transformer_{self.pred_horizon}m_train.pt'
        val_path = self.data_path / f'news_transformer_{self.pred_horizon}m_val.pt'
        test_path = self.data_path / f'news_transformer_{self.pred_horizon}m_test.pt'

        if train_path.exists() and val_path.exists() and test_path.exists() and not force:
            return
        
        news_df = DataSource()
        news_df.create_df('news')

        for path, split_func in zip(
            [train_path, val_path, test_path],
            [self._get_train_split, self._get_val_split, self._get_test_split]
        ):
            split = split_func(news_df.df)

            embeddings = torch.from_numpy(np.stack(split['embeddings'].values).astype(np.float32))
            timestamps = torch.from_numpy(split['elapsed_time'].values.astype(np.float32))

            torch.save({
                'embeddings': embeddings,
                'timestamps': timestamps
            }, path)

            print(
                f'Saved dataset to {path}:\n'
                f'Embeddings: {embeddings.shape},\n'
                f'Timestamps: {timestamps.shape}.\n\n'
            )
    
    def build_dataset(self, force=False):

        if self.transformer:
            self._build_stock_transformer_data(force)

            if self.news:
                self._build_news_transformer_data(force)
    
    def _make_dataset(self, split):

        if self.transformer:
            stock_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_{split}.pt'

            if self.news:
                news_path = self.data_path / f'news_transformer_{self.pred_horizon}m_{split}.pt'
                return StockNewsTransformerDataset(
                    stock_path, news_path, self.stock_lookback,
                    self.pred_horizon, self.time_vec_input
                )
            
            else:
                return StockTransformerDataset(stock_path, self.stock_lookback)
    
    def train(
        self,
        num_epochs,
        batch_size=32,
        lr=1e-3,
        val_every=500,
        ckpt_path=None
    ):
        path = ckpt_path if ckpt_path else self.experiment_path / 'ckpt.pt'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        loaders = {
            split: DataLoader(
                self._make_dataset(split),
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_fn
            )
            for split in ('train', 'val', 'test')
        }

        model = self.model.to(device)
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
                
                print(f"Epoch {epoch + 1}/{num_epochs} ({len(loaders['train']) * epoch} batches).")

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
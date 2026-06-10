import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys
from pathlib import Path
import zarr
from zarr.storage import ZipStore

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from models import StockTransformer
from processing import DataSource, get_stocks, get_text_window

def create_sequences(arr, window_size):
    sequences = []
    num_samples = len(arr)
    for i in range(num_samples - window_size + 1):
        sequences.append(arr[i:i + window_size])
    return np.array(sequences)

def collate_fn(batch):
    args = list(zip(*batch))
    n = len(args)

    for i, arg in enumerate(args[:n]):
        lengths = torch.tensor([len(f) for f in arg])
        if not (lengths == lengths[0]).all():
            args[i] = pad_sequence(arg, batch_first=True, padding_value=0.0)
            B, L_max = args[i].shape[:2]
            arg_mask = torch.arange(L_max).unsqueeze(0) < lengths.unsqueeze(1)
            args.insert(-1, arg_mask)
        else:
            args[i] = torch.stack(list(arg))

    return tuple(args)

class StockTransformerDataset(Dataset):
    def __init__(self, path):
        self.stock_data = zarr.open_group(path, mode='r')
    
    def __len__(self):
        return self.stock_data['features'].shape[0]
    
    def __getitem__(self, idx):
        t = torch.from_numpy(self.stock_data['timestamps'][idx].astype(np.float32))
        x = torch.from_numpy(self.stock_data['features'][idx].astype(np.float32))
        y = torch.from_numpy(self.stock_data['target'][idx].astype(np.float32))
        m = torch.from_numpy(self.stock_data['mask'][idx].astype(np.float32))

        return t, x, m, y

class StockNewsTransformerDataset(StockTransformerDataset):
    def __init__(self, stock_path, news_path, pred_horizon, time_vec_input):
        super().__init__(stock_path)
        self.news_data = zarr.open_group(news_path, mode='r')

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

        non_train_split = df.iloc[last_train_idx - self.window_size + 2:]
        non_test_mask = non_train_split.index.get_level_values('local_time') <= self.val_cutoff

        return non_train_split.loc[non_test_mask]

    def _get_test_split(self, df):
        last_val_idx = df.index.get_loc(self.val_cutoff)

        return df.iloc[last_val_idx - self.window_size + 2:]
    
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

        train_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_train.zarr.zip'
        val_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_val.zarr.zip'
        test_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_test.zarr.zip'

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
        
        train_size = self._get_train_split(self.stock_dfs[0].df).shape[0] - self.stock_lookback + 1
        print(f'Train cutoff: {self.train_cutoff}; no. of training sequences: {train_size}')

        val_size = self._get_val_split(self.stock_dfs[0].df).shape[0] - self.stock_lookback + 1
        print(f'Validation set cutoff: {self.val_cutoff}; no. of validation sequences: {val_size}')

        test_size = self._get_test_split(self.stock_dfs[0].df).shape[0] - self.stock_lookback + 1
        print(f'No. of test sequences: {test_size}')

        chunk_size = 500

        for path, size in zip([train_path, val_path, test_path], [train_size, val_size, test_size]):

            store = ZipStore(path, mode='w')
            root = zarr.open_group(store=store)

            X_shape, y_shape, ts_shape, m_shape = self._get_stock_shapes(size)
            chunk_X_shape, chunk_y_shape, chunk_ts_shape, chunk_m_shape = self._get_stock_shapes(chunk_size)

            zarr_X = root.create_array('features', shape=X_shape, chunks=chunk_X_shape, dtype='float32')
            zarr_y = root.create_array('targets', shape=y_shape, chunks=chunk_y_shape, dtype='float32')
            zarr_ts = root.create_array('timestamps', shape=ts_shape, chunks=chunk_ts_shape, dtype='float32')
            zarr_m = root.create_array('mask', shape=m_shape, chunks=chunk_m_shape, dtype='float32')

            for i in range(0, size, chunk_size):
                end_idx = min(i + chunk_size, size)
                
                chunk_X = []
                chunk_y = []
                chunk_ts = []
                chunk_m = []

                for stock_df in self.stock_dfs:

                    split = None
                    if path == train_path:
                        split = self._get_train_split(stock_df.df)
                    elif path == val_path:
                        split = self._get_val_split(stock_df.df)
                    elif path == test_path:
                        split = self._get_test_split(stock_df.df)
                    
                    chunk_X.append(create_sequences(
                        split[stock_df.features].iloc[i:end_idx + self.stock_lookback - 1].values,
                        self.stock_lookback
                    ))

                    target = split[stock_df.target].iloc[i + self.stock_lookback - 1:end_idx + self.stock_lookback - 1].values
                    chunk_y.append(np.array([target, 1 - target]))
                    
                    chunk_ts.append(create_sequences(
                        split[stock_df.df[stock_df.time_vec_input]].iloc[i:end_idx + self.stock_lookback - 1].values,
                        self.stock_lookback
                    ))

                    chunk_m.append(create_sequences(
                        split[stock_df.no_activity_col].iloc[i:end_idx + self.stock_lookback - 1].values,
                        self.stock_lookback
                    ))

                chunk_X = np.transpose(np.array(chunk_X), (1, 0, 2, 3))
                chunk_y = np.transpose(np.array(chunk_y), (2, 0, 1))
                chunk_ts = np.transpose(np.array(chunk_ts), (1, 0, 2))
                chunk_m = np.transpose(np.array(chunk_m), (1, 0, 2))

                zarr_X[i:end_idx] = chunk_X
                zarr_y[i:end_idx] = chunk_y
                zarr_ts[i:end_idx] = chunk_ts
                zarr_m[i:end_idx] = chunk_m
                
                print(f"Saved sequences {i} to {end_idx} safely to disk.")
    
    def _build_news_transformer_data(self, force=False):

        train_path = self.data_path / f'news_transformer_{self.pred_horizon}m_train.zarr.zip'
        val_path = self.data_path / f'news_transformer_{self.pred_horizon}m_val.zarr.zip'
        test_path = self.data_path / f'news_transformer_{self.pred_horizon}m_test.zarr.zip'

        if train_path.exists() and val_path.exists() and test_path.exists() and not force:
            return
        
        news_df = DataSource()
        news_df.create_df('news')

        news_train = self._get_train_split(news_df.df)
        news_val = self._get_val_split(news_df.df)
        news_test = self._get_test_split(news_df.df)

        train_arr = np.stack(news_train['embeddings'].values)
        train_t = news_train['elapsed_time'].values
        val_arr = np.stack(news_val['embeddings'].values)
        val_t = news_val['elapsed_time'].values
        test_arr = np.stack(news_test['embeddings'].values)
        test_t = news_test['elapsed_time'].values

        z = zarr.open(train_path, mode='w')
        z['embeddings'] = train_arr
        z['timestamps'] = train_t

        z = zarr.open(val_path, mode='w')
        z['embeddings'] = val_arr
        z['timestamps'] = val_t

        z = zarr.open(test_path, mode='w')
        z['embeddings'] = test_arr
        z['timestamps'] = test_t
    
    def build_dataset(self, force=False):

        if self.transformer:
            self._build_stock_transformer_data(force)

            if self.news:
                self._build_news_transformer_data(force)
    
    def _make_dataset(self, split):

        if self.transformer:
            stock_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_{split}.zarr.zip'

            if self.news:
                news_path = self.data_path / f'news_transformer_{self.pred_horizon}m_{split}.zarr.zip'
                return StockNewsTransformerDataset(stock_path, news_path, self.pred_horizon, self.time_vec_input)
            
            else:
                return StockTransformerDataset(stock_path)
    
    def train(
        self,
        num_epochs,
        batch_size=32,
        lr=1e-3,
    ):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        loaders = {
            split: DataLoader(
                self._make_dataset(split),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn
            )
            for split in ('train', 'val', 'test')
        }

        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for *args, target in loaders['train']:
                target = target.argmax(dim=-1)       # (B, 30, 2) → (B, 30)

                target   = target.to(device)
                for arg in args:
                    arg = arg.to(device)

                optimizer.zero_grad()
                logits = model(*args)       # (B, 30, 2)
                logits = logits.permute(0, 2, 1)     # (B, 2, 30)

                loss = criterion(logits, target)     # target (B, 30)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loaders['train'])
            print(f"Epoch {epoch + 1}/{num_epochs} train loss: {avg_loss:.4f}")
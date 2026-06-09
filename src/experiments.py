import numpy as np
import torch
from torch.utils.data import Dataset
import sys
from pathlib import Path
import zarr

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from models import StockTransformer
from processing import DataSource, get_stocks

def get_train_split(df, train_cutoff):
    return df.loc[df.index.get_level_values('local_time') <= train_cutoff]

def get_val_split(df, train_cutoff, val_cutoff, window_size):
    last_train_idx = df.index.get_loc(train_cutoff)

    non_train_split = df.iloc[last_train_idx - window_size + 2:]
    non_test_mask = non_train_split.index.get_level_values('local_time') <= val_cutoff

    return non_train_split.loc[non_test_mask]

def get_test_split(df, val_cutoff, window_size):
    last_val_idx = df.index.get_loc(val_cutoff)

    return df.iloc[last_val_idx - window_size + 2:]

def create_sequences(arr, window_size):
    sequences = []
    num_samples = len(arr)
    for i in range(num_samples - window_size + 1):
        sequences.append(arr[i:i + window_size])
    return np.array(sequences)

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
        self.pred_30 = pred_30
        self.news = news
        self.social = social
        self.stock_lookback = stock_lookback
        self.processed_path = Path('data/processed')
        self.experiment_path = Path(f'experiments/{experiment_name}')
        self.experiment_path.mkdir(parents=True, exist_ok=True)
    
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
    
    def build_dataset(self, force=False):
        self.data_path = self.experiment_path / 'data'
        self.data_path.mkdir(parents=True, exist_ok=True)

        stocks = get_stocks()

        if self.transformer:
            train_path = self.data_path / 'stock_train.zarr'
            val_path = self.data_path / 'stock_val.zarr'
            test_path = self.data_path / 'stock_test.zarr'

            if train_path.exists() and val_path.exists() and test_path.exists() and not force:
                return

            self.stock_dfs = []
            for stock in stocks:
                print(f'Loading {stock}...')
                stock_df = DataSource()
                stock_df.create_df(file_name=f'{stock}_{30 if self.pred_30 else 10}m')
                stock_df.df = stock_df.df.sort_index()
                self.stock_dfs.append(stock_df)
            
            train_cutoff = self.stock_dfs[0].train_cutoff
            train_size = get_train_split(
                self.stock_dfs[0].df,
                train_cutoff
            ).shape[0] - self.stock_lookback + 1

            print(f'Train cutoff: {train_cutoff}; no. of training sequences: {train_size}')

            filtered_date_times = self.stock_dfs[0].filtered_date_times
            val_cutoff = filtered_date_times[int(0.9 * len(filtered_date_times))]
            val_size = get_val_split(
                self.stock_dfs[0].df,
                train_cutoff,
                val_cutoff,
                self.stock_lookback
            ).shape[0] - self.stock_lookback + 1

            print(f'Validation set cutoff: {val_cutoff}; no. of validation sequences: {val_size}')

            test_size = get_test_split(
                self.stock_dfs[0].df,
                val_cutoff,
                self.stock_lookback
            ).shape[0] - self.stock_lookback + 1

            print(f'No. of test sequences: {test_size}')

            chunk_size = 500

            for path, size in zip([train_path, val_path, test_path], [train_size, val_size, test_size]):

                root = zarr.open_group(store=path, mode='w')

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
                            split = get_train_split(stock_df.df, train_cutoff)
                        elif path == val_path:
                            split = get_val_split(stock_df.df, train_cutoff, val_cutoff, self.stock_lookback)
                        elif path == test_path:
                            split = get_test_split(stock_df.df, val_cutoff, self.stock_lookback)
                        
                        chunk_X.append(create_sequences(
                            split[stock_df.features].iloc[i:end_idx + self.stock_lookback - 1].values,
                            self.stock_lookback
                        ))

                        target = split[stock_df.target].iloc[i + self.stock_lookback - 1:end_idx + self.stock_lookback - 1].values
                        chunk_y.append(np.array([target, 1 - target]))
                        
                        chunk_ts.append(create_sequences(
                            split[stock_df.time_vec_input].iloc[i:end_idx + self.stock_lookback - 1].values,
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
                    
                    print(f"Saved rows {i} to {end_idx} safely to disk.")
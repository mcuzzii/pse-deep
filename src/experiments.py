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
    
    def build_dataset(self, force=False):
        self.data_path = self.experiment_path / 'data'
        self.data_path.mkdir(parents=True, exist_ok=True)

        stocks = get_stocks()

        if self.transformer:
            if (self.data_path / 'stocks.zarr').exists() and not force:
                return
            
            store = zarr.DirectoryStore(self.data_path / 'stocks.zarr')
            root = zarr.group(store=store, overwrite=True)

            stock_dfs = []
            train_cutoffs = set()
            for stock in stocks:
                stock_df = DataSource()
                stock_df.create_df(file_name=f'{stock}_{30 if self.pred_30 else 10}m')
                stock_dfs.append(stock_df)
                train_cutoffs.add(stock_df.train_cutoffs)
            
            train_cutoff = None
            if len(train_cutoffs) != 1:
                raise ValueError("Train cutoffs do not align across stocks.")
            else:
                train_cutoff = next(iter(train_cutoffs))
            

            zarr_X = root.create_dataset('features', shape=x_shape, chunks=(500, 30, 60, 100), dtype='float32')
            zarr_y = root.create_dataset('targets', shape=y_shape, chunks=(500, 30, 2), dtype='float32')
            zarr_ts = root.create_dataset('timestamps', shape=ts_shape, chunks=(500, 30, 60), dtype='float32')
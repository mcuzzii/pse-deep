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
import math
import random
import matplotlib.pyplot as plt

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from models import StockTransformer, StockNewsTransformer, StockSocialTransformer, StockNewsSocialTransformer
from processing import DataSource, get_stocks, get_text_window, get_elapsed_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn(
    input_dim: int | None = None,
    K: int | None = None
):
    def func(batch):
        args = list(zip(*batch))
        n = len(args)

        masks = []

        for i, arg in enumerate(args[:n]):
            if len(arg[0].shape)== 1 or (len(arg[0].shape) == 2 and arg[0].shape[1] == input_dim):
                args[i] = pad_sequence(arg, batch_first=True, padding_value=0.0)

                # arg.shape = (B, Tn) or (B, Tn, En)
                if args[i].shape[1] < K:
                    pad_shape = (args[i].shape[0], K - args[i].shape[1], *args[i].shape[2:])
                    zeros = torch.zeros(pad_shape, dtype=args[i].dtype, device=args[i].device)
                    args[i] = torch.cat([args[i], zeros], dim=1)

                if len(args[i].shape) == 3:
                    lengths = torch.tensor([len(f) for f in arg])
                    L_max = args[i].shape[1]
                    arg_mask = torch.arange(L_max).unsqueeze(0) < lengths.unsqueeze(1)

                    masks.append(arg_mask)
            
            else:
                args[i] = torch.stack(list(arg))
        
        for mask in masks:
            args.insert(-1, mask)

        return tuple(args)
    return func

def _run_validation(model, loaders, device, criterion):
    model.eval()
    total_val_loss = 0

    with tqdm(total=len(loaders['val']), desc="Validation") as pbar:

        with torch.no_grad():
            for *args, target in loaders['val']:
                target = target.argmax(dim=-1).to(device)

                args = [a.to(device) for a in args]

                logits = model(*args)
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

        # print(f'Shapes: X: {x.shape}; y: {y.shape}; ts: {t.shape}')

        return t, x, y

class StockNewsTransformerDataset(StockTransformerDataset):
    def __init__(self, stock_path, news_path, stock_lookback, pred_horizon, time_vec_input):
        super().__init__(stock_path, stock_lookback)
        self.news_data = torch.load(news_path)

        self.pred_horizon = pred_horizon
        self.time_vec_input = time_vec_input
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        t, x, y = super().__getitem__(idx)

        self.last_timestamp = float(t[0, -1])

        idx = (self.time_vec_input - self.last_timestamp).abs().idxmin()
        cutoff, _ = get_text_window(idx, self.time_vec_input.index, self.pred_horizon)
        self.cutoff_scaled = get_elapsed_time(cutoff)

        embeddings = self.news_data['embeddings']
        timestamps = self.news_data['timestamps']
        self.window = (self.cutoff_scaled < timestamps) & (timestamps <= self.last_timestamp)

        # print(f'Shapes: news_e: {embeddings[window].shape}; news_t: {timestamps[window].shape}')

        return t, timestamps[self.window], x, embeddings[self.window], y

class StockSocialTransformerDataset(StockNewsTransformerDataset):
    def __init__(self, stock_path, social_path, stock_lookback, pred_horizon, time_vec_input):
        super().__init__(stock_path, social_path, stock_lookback, pred_horizon, time_vec_input)

        self.social_data = self.news_data
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        t, ts, x, es, y = super().__getitem__(idx)

        impact = self.social_data['impact']
        
        return t, ts, x, impact[self.window], es, y

class StockNewsSocialTransformerDataset(StockNewsTransformerDataset):
    def __init__(
        self,
        stock_path,
        social_path,
        news_path,
        stock_lookback,
        pred_horizon,
        time_vec_input
    ):
        super().__init__(stock_path, news_path, stock_lookback, pred_horizon, time_vec_input)
        self.social_data = torch.load(social_path)
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        t, tn, x, en, y = super().__getitem__(idx)

        ts = self.social_media['timestamps']

        window = (self.cutoff_scaled < ts) & (ts <= self.last_timestamp)

        s = self.social_media['impact'][window]
        ts = ts[window]
        es = self.social_media['embeddings'][window]

        return t, tn, ts, x, s, en, es


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False

    def __call__(self, val_loss):
        # Check if the validation loss improved significantly
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset patience counter
            return True       # Signals a new best model found
        else:
            self.counter += 1
            print(f"\nEarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
            return False      # No improvement

class SigmaAnnealer:
    def __init__(self, model, sigma_start=0.05, sigma_end=1e-5, num_batches=500):

        self.null = False
        if not isinstance(model, (
            StockNewsTransformer,
            StockSocialTransformer,
            StockNewsSocialTransformer
        )):
            self.null = True
            return
        
        self.topk = model.topk
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.num_batches = num_batches

    def __call__(self, batches: int):

        if self.null:
            return

        t = batches / self.num_batches
        sigma = self.sigma_start * (self.sigma_end / self.sigma_start) ** t
        self.topk.set_sigma(sigma)
        return sigma

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
        self.experiment_name = experiment_name

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
        return df.df.loc[df.df.index.get_level_values('local_time') <= self.train_cutoff]

    def _get_val_split(self, df):
        last_train_idx = self.filtered_date_times.get_loc(self.train_cutoff)

        if df.file_name in ('news', 'social'):
            cutoff, _ = get_text_window(
                self.time_vec_input.index[last_train_idx + 1],
                self.filtered_date_times,
                self.pred_horizon,
                24
            )
            non_train_split = df.df.loc[df.df.index.get_level_values('local_time') > cutoff]

        else:
            non_train_split = df.df.iloc[last_train_idx - self.stock_lookback + 2:]

        non_test_mask = non_train_split.index.get_level_values('local_time') <= self.val_cutoff
        return non_train_split.loc[non_test_mask]

    def _get_test_split(self, df):
        last_val_idx = self.filtered_date_times.get_loc(self.val_cutoff)

        if df.file_name in ('news', 'social'):
            cutoff, _ = get_text_window(
                self.time_vec_input.index[last_val_idx + 1],
                self.filtered_date_times,
                self.pred_horizon,
                24
            )

            return df.df.loc[df.df.index.get_level_values('local_time') > cutoff]
        
        else:
            return df.df.iloc[last_val_idx - self.stock_lookback + 2:]
    
    def build_model(
        self,
        input_dim: int,
        social_input_dim: int | None = None,
        text_input_dim: int | None = None,
        social_embedding_dim: int | None = None,
        hidden_dim: int | None = None,
        embedding_dim: int | None = None,
        temporal_embedding_dim: int | None = None,
        K=30,
        num_samples=100,
        sigma=5e-2,
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
        elif self.transformer and self.news and not self.social:
            self.model = StockNewsTransformer(
                input_dim,
                text_input_dim,
                embedding_dim,
                temporal_embedding_dim,
                num_heads,
                K,
                num_samples,
                sigma,
                num_layers,
                expansion,
                dropout
            )
        elif self.transformer and self.social and not self.news:
            self.model = StockSocialTransformer(
                input_dim,
                social_input_dim,
                text_input_dim,
                social_embedding_dim,
                embedding_dim,
                temporal_embedding_dim,
                num_heads,
                K,
                num_samples,
                sigma,
                num_layers,
                expansion,
                dropout
            )
        elif self.transformer and self.social and self.news:
            self.model = StockNewsSocialTransformer(
                input_dim,
                social_input_dim,
                text_input_dim,
                social_embedding_dim,
                embedding_dim,
                temporal_embedding_dim,
                num_heads,
                K,
                num_samples,
                sigma,
                num_layers,
                expansion,
                dropout
            )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
    
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
                'target': list()
            }

            for stock_df in self.stock_dfs:

                split = split_func(stock_df)
                target = split[stock_df.target].iloc[self.stock_lookback - 1:].values

                stock_X = split[stock_df.features].values
                stock_y = np.array([target, 1 - target])
                stock_ts = split[stock_df.time_vec_input].values
                
                tensors['features'].append(stock_X)
                tensors['target'].append(stock_y)
                tensors['timestamps'].append(stock_ts)

                print(
                    f'Processed {stock_df.file_name} into arrays:\n'
                    f'- Timestamps: {stock_ts.shape},\n'
                    f'- Features: {stock_X.shape},\n'
                    f'- Target: {stock_y.shape}.\n\n'
                )

            tensors['features'] = torch.from_numpy(np.array(tensors['features'], dtype=np.float32))         # (30, B, 100)
            tensors['target'] = torch.from_numpy(np.array(tensors['target'], dtype=np.float32))          # (30, 2, B)
            tensors['timestamps'] = torch.from_numpy(np.array(tensors['timestamps'], dtype=np.float32))     # (30, B)

            print(
                'Final dataset sizes:\n'
                f'- Timestamps: {tensors['timestamps'].shape},\n'
                f'- Features: {tensors['features'].shape},\n'
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
            split = split_func(news_df)

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
    
    def _build_social_transformer_data(self, force=False):

        train_path = self.data_path / f'social_transformer_{self.pred_horizon}m_train.pt'
        val_path = self.data_path / f'social_transformer_{self.pred_horizon}m_val.pt'
        test_path = self.data_path / f'social_transformer_{self.pred_horizon}m_test.pt'

        if train_path.exists() and val_path.exists() and test_path.exists() and not force:
            return
        
        social_df = DataSource()
        social_df.create_df('social_media')

        social_features = DataSource()
        social_features.create_df(f'social_media_{self.pred_horizon}m')

        selected_features = set(social_features.selected_features)

        keywords = {
            'retweet_count',
            'reply_count',
            'like_count',
            'quote_count',
            'view_count',
            'bookmark_count',
            'author_is_blue_verified',
            'author_followers',
            'author_following',
            'author_favourites_count',
            'author_media_count',
            'author_statuses_count'
        }

        impact_features = set()
        for key in keywords:
            if any(s.startswith(key) for s in selected_features):
                impact_features.add(key)
        if any('follower_weighted_mean' in s for s in selected_features):
            impact_features.add('author_followers')
        if any('viral_coeff' in s for s in selected_features):
            impact_features.add('reply_count')
        
        impact_features = list(impact_features)
        print(f'Impact features: {impact_features}')

        for path, split_func in zip(
            [train_path, val_path, test_path],
            [self._get_train_split, self._get_val_split, self._get_test_split]
        ):
            split = split_func(social_df)

            embeddings = torch.from_numpy(np.stack(split['embeddings'].values).astype(np.float32))
            impact = torch.from_numpy(split[impact_features].values.astype(np.float32))
            timestamps = torch.from_numpy(split['elapsed_time'].values.astype(np.float32))

            torch.save({
                'embeddings': embeddings,
                'impact': impact,
                'timestamps': timestamps
            }, path)

            print(
                f'Saved dataset to {path}:\n'
                f'Embeddings: {embeddings.shape},\n'
                f'Impact: {impact.shape},\n'
                f'Timestamps: {timestamps.shape}.\n\n'
            )
    
    def build_dataset(self, force=False):

        if self.transformer:
            self._build_stock_transformer_data(force)

            if self.news:
                self._build_news_transformer_data(force)
            
            if self.social:
                self._build_social_transformer_data(force)
    
    def _make_dataset(self, split):

        if self.transformer:
            stock_path = self.data_path / f'stock_transformer_{self.pred_horizon}m_{split}.pt'

            if self.news:
                news_path = self.data_path / f'news_transformer_{self.pred_horizon}m_{split}.pt'

                if self.social:
                    social_path = self.data_path / f'social_transformer_{self.pred_horizon}m_{split}.pt'

                    return StockNewsSocialTransformerDataset(
                        stock_path, social_path, news_path, self.stock_lookback,
                        self.pred_horizon, self.time_vec_input
                    )
                
                return StockNewsTransformerDataset(
                    stock_path, news_path, self.stock_lookback,
                    self.pred_horizon, self.time_vec_input
                )

            elif self.social:
                social_path = self.data_path / f'social_transformer_{self.pred_horizon}m_{split}.pt'

                return StockSocialTransformerDataset(
                    stock_path, social_path, self.stock_lookback,
                    self.pred_horizon, self.time_vec_input
                )
            
            return StockTransformerDataset(stock_path, self.stock_lookback)
    
    def train(
        self,
        num_epochs,
        batch_size=32,
        accumulation_steps=1,
        lr=1e-5,
        val_every=1024,
        patience=10,
        weight_decay=1e-2,
        sigma_end=1e-5
    ):
        interrupted = False

        def handler(sig, frame):
            nonlocal interrupted
            interrupted = True
            print("Interrupt received. Saving checkpoint...")

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        path = self.experiment_path / 'checkpoints' / f'{self.experiment_name}.pt'
        best_path = self.experiment_path / f'{self.experiment_name}.pt' # <-- Target path for best weights
        path.parent.mkdir(parents=True, exist_ok=True)

        model = self.model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()

        loaders = {
            split: DataLoader(
                self._make_dataset(split),
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=2,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(42),
                pin_memory=True,
                collate_fn=collate_fn(input_dim=getattr(model, 'text_embedding_dim', None), K=getattr(model, 'K', None))
            )
            for split in ('train', 'val', 'test')
        }

        global_step = 0
        num_batches = len(loaders['train']) * num_epochs
        sigma_start = getattr(model, 'sigma', 5e-2)

        if isinstance(val_every, int):
            val_every = lambda x: val_every * x
        val_generator = lambda x: accumulation_steps * math.ceil(val_every(x) / accumulation_steps)
        
        self.val_periods = [
            val_generator(x)
            for x in range(num_batches)
            if val_generator(x) < num_batches
        ]
        self.val_periods.append(num_batches)

        resume_step = None
        if path.exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            
            resume_step = checkpoint["global_step"]
            class_weights = checkpoint["class_weights"]
            train_losses = checkpoint["train_losses"]
            val_losses = checkpoint["val_losses"]
            total_loss = checkpoint["total_loss"]
            early_stopper = checkpoint["early_stopper"]
            sigma_annealer_args = checkpoint["sigma_annealer"]

            if early_stopper.stop:
                print(f"Model already saved in {best_path}. Skipping..")
                return
        
        else:
            resume_step = None
            train_losses = []
            val_losses = []
            total_loss = 0
            early_stopper = EarlyStopping(patience=patience)
            sigma_annealer_args = {
                'sigma_start': sigma_start,
                'sigma_end': sigma_end,
                'num_batches': num_batches
            }

            print("Calculating class weights from training set...")
            train_dataset = loaders['train'].dataset
            
            all_targets = []
            for i in range(min(len(train_dataset), 5000)): # Scan a large sample or full dataset
                *_, tgt = train_dataset[i]
                all_targets.append(torch.tensor(tgt).argmax(dim=-1))
            
            flat_targets = torch.cat(all_targets)
            count_0 = (flat_targets == 0).sum().item()
            count_1 = (flat_targets == 1).sum().item()
            total_counts = count_0 + count_1
            
            weight_0 = total_counts / (2.0 * count_0)
            weight_1 = total_counts / (2.0 * count_1)
            
            class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float, device=device)
            print(f"Computed Class Weights: Class 0: {weight_0:.4f}, Class 1: {weight_1:.4f}")
        
        accumulation_loss = 0

        sigma_annealer = SigmaAnnealer(model, **sigma_annealer_args)
        pbar = tqdm(total=num_batches, desc="Training")

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training loop
        try:
            for epoch in range(num_epochs):
                model.train()

                for i, (*args, target) in enumerate(loaders['train']):

                    if interrupted:
                        global_step = accumulation_steps * (global_step // accumulation_steps)
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

                    logits = model(*args)                # (B, 30, 2)
                    logits = logits.permute(0, 2, 1)     # (B, 2, 30)

                    loss = criterion(logits, target)     # target (B, 30)
                    loss = loss / accumulation_steps
                    loss.backward()
                    accumulation_loss += loss.item() * accumulation_steps

                    if (i + 1) % accumulation_steps == 0:
                        total_loss += accumulation_loss
                        accumulation_loss = 0
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    global_step += 1
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())

                    if global_step in self.val_periods:
                        val_loss = _run_validation(model, loaders, device, criterion)
                        val_losses.append(val_loss)

                        period_idx = self.val_periods.index(global_step)
                        num_steps = (global_step - self.val_periods[period_idx - 1]) if period_idx > 0 else global_step

                        train_loss = total_loss / num_steps
                        train_losses.append(train_loss)

                        print(f'train_loss: {train_loss}, val_loss: {val_loss}')

                        is_best = early_stopper(val_loss)
                        sigma_annealer(global_step)

                        checkpoint_data = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "global_step": global_step,
                            "class_weights": class_weights,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "total_loss": total_loss,
                            "early_stopper": early_stopper,
                            "sigma_annealer": {
                                'sigma_start': sigma_start,
                                'sigma_end': sigma_end,
                                'num_batches': num_batches
                            }
                        }

                        # optional checkpoint on validation
                        torch.save(checkpoint_data, path)

                        if is_best:
                            print(f"\nNew best validation loss achieved ({val_loss:.4f})! Saving best model weights...")
                            torch.save(checkpoint_data, best_path)

                        total_loss = 0

                        if early_stopper.stop:
                            print("\nEarly stopping triggered. Training halted.")
                            interrupted = True
                            break
                
                if (i + 1) % accumulation_steps != 0:
                    total_loss += accumulation_loss
                    accumulation_loss = 0
                    optimizer.step()
                    optimizer.zero_grad()
                
                if interrupted:
                    break
                
                print(f"Epoch {epoch + 1}/{num_epochs}.")
            
            early_stopper.stop = True

        except KeyboardInterrupt:
            pass

        finally:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "class_weights": class_weights,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "total_loss": total_loss,
                "early_stopper": early_stopper,
                "sigma_annealer": {
                    'sigma_start': sigma_start,
                    'sigma_end': sigma_end,
                    'num_batches': num_batches
                }
            }, path)
            
            pbar.close()
            self.plot_loss_curves()
    
    def plot_loss_curves(self):
        model = torch.load(
            self.experiment_path / 'checkpoints' / f'{self.experiment_name}.pt',
            map_location=device,
            weights_only=False
        )
        train_losses = model['train_losses']
        val_losses = model['val_losses']

        x = self.val_periods[:len(train_losses)]
        
        topk = getattr(model, 'topk', None)

        if topk:
            print(topk.sigma)

        plt.figure(figsize=(8, 5))

        plt.plot(x, train_losses, label="Train Loss")
        plt.plot(x, val_losses, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()

        save_path = self.experiment_path / 'loss_curve.png'

        plt.savefig(save_path, dpi=300)
        plt.close()
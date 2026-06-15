import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path.cwd() / 'src'))

from experiments import Experiment, collate_fn

stock_news_transformer = Experiment(
    experiment_name='stock_news_transformer_30',
    transformer=True,
    pred_30=True,
    news=True,
    social=False,
    stock_lookback=60
)
stock_news_transformer.build_dataset()

loaders = {
    split: DataLoader(
        stock_news_transformer._make_dataset(split),
        batch_size=8,
        shuffle=(split == 'train'),
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    for split in ('train', 'val', 'test')
}

counter = 0
for i, (*args, target) in enumerate(loaders['train']):
    print(args[0][:, 0, :])
    counter += 1
    if counter == 100:
        break
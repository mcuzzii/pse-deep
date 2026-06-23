import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from river.drift import ADWIN
import numpy as np
from tqdm import tqdm

class Eval:
    def __init__(self):
        self.experiments_path = Path('experiments')
        self.results_path = self.experiments_path / 'results'
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def overall_metrics(self):
        metrics = dict()

        for dir in self.experiments_path.iterdir():

            if dir.name in ('data', 'experiments', 'results'):
                continue

            test_outputs = dir / 'test_outputs.pt'
            print(f'Loading {test_outputs}...')

            metrics_dict = {
                k: v
                for k, v in torch.load(test_outputs, map_location=torch.device('cpu'), weights_only=False).items()
                if not k.endswith('all_targets') and not k.endswith('logit_scores')
            }

            metrics[dir.name] = metrics_dict
        
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df.to_csv(self.results_path / 'model_scores.csv')

        return df
    
    def compute_model_drift(self):
        for dir in self.experiments_path.iterdir():

            if dir.name in ('data', 'experiments', 'results'):
                continue

            test_outputs = dir / 'test_outputs.pt'
            out = torch.load(test_outputs, map_location=torch.device('cpu'), weights_only=False)

            logit_scores = out['test_logit_scores']                         # N, S, 2 for transformers, N*S, 1, 2 for mlp
            targets = out['test_all_targets']                               # N, S for transformers, N*S, 1 for mlp
            if 'mlp' in dir.name:
                logit_scores = logit_scores.squeeze(1).reshape(30, logit_scores.shape[0] // 30, -1).transpose(0, 1)
                targets = targets.squeeze(1).reshape(30, targets.shape[0] // 30).transpose(0, 1)

            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(logit_scores.permute(0, 2, 1), targets)        # N, S

            detectors = [ADWIN() for _ in range(loss.shape[1])]
            windows = [[] for _ in range(loss.shape[1])]
            means = torch.zeros_like(loss)
            width_histories = [[] for _ in range(loss.shape[1])]

            for s in range(loss.shape[1]):
                for n in tqdm(range(loss.shape[0])):
                    val = loss[n, s].item()
                    detectors[s].update(val)

                    print(-detectors[s].width)

                    windows[s].append(val)
                    windows[s] = windows[s][-detectors[s].width:]
                    means[n, s] = sum(windows[s]) / len(windows[s])
                    width_histories[s].append(detectors[s].width)

            drift_scores = ((loss - means) ** 2).mean(dim=0)
            drift_scores_norm = (drift_scores - drift_scores.min()) / (drift_scores.max() - drift_scores.min())
            avg_widths = torch.tensor([sum(w) / len(w) for w in width_histories])
            avg_widths_norm = (avg_widths - avg_widths.min()) / (avg_widths.max() - avg_widths.min())
            drift_from_width = 1 - avg_widths_norm

            print(f'Drift scores: {drift_scores_norm}')
            print(f'Drift from width: {drift_from_width}')

            print(f'Corr: {np.corrcoef(drift_scores_norm.numpy(), drift_from_width.numpy())}')
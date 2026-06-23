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
        model_scores = pd.read_csv(self.results_path / 'model_scores.csv', index_col=0)

        overall_model_drift_scores = dict()
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
            width_histories = torch.zeros_like(loss)

            for s in tqdm(range(loss.shape[1])):
                for n in range(loss.shape[0]):
                    val = loss[n, s].item()
                    detectors[s].update(val)

                    windows[s].append(val)
                    windows[s] = windows[s][int(-detectors[s].width):]
                    means[n, s] = sum(windows[s]) / len(windows[s])
                    width_histories[n, s] = detectors[s].width
            
            msd = ((loss - means) ** 2)

            msd = (msd - msd.min()) / (msd.max() - msd.min())
            mean_squared_loss_deviations = msd.mean(dim=0)

            width_histories = (width_histories - width_histories.min()) / (width_histories.max() - width_histories.min())
            drift_from_width = 1 - width_histories.mean(dim=0)

            msd_mean = msd.mean().item()
            widths_mean = 1 - width_histories.mean().item()
            combined_drift_score_mean = msd_mean * widths_mean

            print(f'Drift scores: {mean_squared_loss_deviations}')
            print(f'Drift from width: {drift_from_width}')

            out['mean_squared_loss_deviations'] = mean_squared_loss_deviations
            out['drift_from_width'] = drift_from_width
            out['combined_drift_scores'] = mean_squared_loss_deviations * drift_from_width

            torch.save(out, test_outputs)

            overall_model_drift_scores[dir.name] = {
                'msd_mean': msd_mean,
                'widths_mean': widths_mean,
                'combined_drift_score_mean': combined_drift_score_mean
            }
        
        overall_model_drift_scores = pd.DataFrame.from_dict(overall_model_drift_scores, orient='index')
        model_scores = model_scores.join(overall_model_drift_scores, how='left')
        model_scores.to_csv(self.results_path / 'model_scores.csv')
    

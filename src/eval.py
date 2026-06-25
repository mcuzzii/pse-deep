import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from river.drift import ADWIN
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score
)

sys.path.append(str(Path.cwd() / 'src'))

from experiments import mcc_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                if (
                    'accuracy' in k or
                    'mcc' in k or
                    'precision' in k or
                    'recall' in k or
                    'f1' in k or
                    'avg_loss' in k
                )
            }

            metrics[dir.name] = metrics_dict
        
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df.to_csv(self.results_path / 'model_scores.csv')

        return df
    
    def compute_experiment_data(self):
        model_scores = pd.read_csv(self.results_path / 'model_scores.csv', index_col=0)

        overall_scores = dict()
        for dir in self.experiments_path.iterdir():

            if dir.name in ('data', 'experiments', 'results'):
                continue

            model_path = dir / f'{dir.name}.pt'
            model = torch.load(model_path, map_location=device, weights_only=False)

            test_outputs = dir / 'test_outputs.pt'
            out = torch.load(test_outputs, map_location=device, weights_only=False)

            val_calib_thresholds = model['best_threshold']                                  # (S,)
            logit_scores = out['test_logit_scores']
            softmax_scores = torch.softmax(logit_scores, dim=-1)[..., -1]                   # N, S, 2 -> N, S
            targets = out['test_all_targets']                                               # N, S

            best_thresholds = torch.zeros_like(targets)
            for i in range(0, logit_scores.shape[0], (logit_scores.shape[0] // 10 + 1)):
                start_idx = i
                end_idx = min(i + logit_scores.shape[0] // 10 + 1, logit_scores.shape[0])

                if i == 0:
                    best_thresholds[start_idx:end_idx] = val_calib_thresholds.unsqueeze(0).expand_as(targets[start_idx:end_idx])
                else:
                    thresholds = mcc_curve(targets[start_idx:end_idx], softmax_scores[start_idx:end_idx])
                    print(f'Best thresholds: {thresholds}')
                    best_thresholds[start_idx:end_idx] = thresholds[-1].unsqueeze(0).expand_as(targets[start_idx:end_idx])
                    print(f'Thresholds tensor: {best_thresholds}')

            preds = softmax_scores >= best_thresholds                                       # N, S
            
            preds_flat = preds.flatten()
            targets_flat = targets.flatten()
            preds_np = preds_flat.cpu().numpy()
            targets_np = targets_flat.cpu().numpy()

            tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()                         # N, S -> S,
            tn = ((preds == 0) & (targets == 0)).sum(dim=0).float()
            fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
            fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()

            numerator = tp * tn - fp * fn                                                   # S,
            denom = torch.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))                       # S,

            mcc = torch.where(denom > 0, numerator / denom, torch.zeros_like(numerator))    # S,
            out['mcc_scores'] = mcc
            print(f'MCCs: {mcc}')
            print()

            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(logit_scores.permute(0, 2, 1), targets)                        # N, S

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

            overall_scores[dir.name] = {
                'test_accuracy_rolling': (preds_flat == targets_flat).float().mean().item(),
                'test_mcc_rolling': matthews_corrcoef(targets_np, preds_np),
                'test_precision_rolling': precision_score(targets_np, preds_np),
                'test_recall_rolling': recall_score(targets_np, preds_np),
                'test_f1_rolling': f1_score(targets_np, preds_np),
                'test_precision_neg_rolling': precision_score(1 - targets_np, 1 - preds_np),
                'test_recall_neg_rolling': recall_score(1 - targets_np, 1 - preds_np),
                'test_f1_neg_rolling': f1_score(1 - targets_np, 1 - preds_np),
                'msd_mean': msd_mean,
                'widths_mean': widths_mean,
                'combined_drift_score_mean': combined_drift_score_mean
            }
        
        overall_scores = pd.DataFrame.from_dict(overall_scores, orient='index')
        model_scores[overall_scores.columns] = overall_scores
        model_scores.to_csv(self.results_path / 'model_scores.csv')
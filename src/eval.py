import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from river.drift import ADWIN

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

            logit_scores = out['test_logit_scores']            # N, S, 2 for transformers, N, 1, 2 for mlp
            if 'mlp' in dir.name:
                logit_scores = logit_scores.squeeze(1).reshape(30, logit_scores.shape[0] // 30, -1).transpose(0, 1)
            
            targets = out['test_all_targets']                  # N, S

            criterion = nn.CrossEntropyLoss(reduction='none')

            loss = criterion(logit_scores.permute(0, 2, 1), targets) # N, S

            for s in loss.shape[1]:

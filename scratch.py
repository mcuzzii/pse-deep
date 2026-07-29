import joblib
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource
from eval import Eval
import torch

summary_tensors = torch.load(
    'experiments/results/attn_analysis/summary_tensors.pt',
    map_location=torch.device('cpu'),
    weights_only=False
)

evaluator = Eval()
evaluator.main_and_interaction_effects()

for dir in Path('experiments').iterdir():
    if dir.name in ('data', 'results') or 'mlp' in dir.name:
        continue

    pred_30 = '30' in dir.name
    pred_horizon = 30 if pred_30 else 10

    if 'social' in dir.name:
        social_features = DataSource()
        social_features.create_df(f'social_media_{pred_horizon}m')

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
        
        evaluator.impact_features = list(impact_features)

    for category in (
        ['Market Open', 'AM Session', 'PM Session', 'Market Close'] +
        ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] +
        [f'Week {i}' for i in range(1, 8)] +
        ['Overall']
    ):
        for k, v in summary_tensors[dir.name][category].items():
            if k in ('tst', 'sft', 'nft', 'ist'):
                vals = summary_tensors[dir.name][category][k].cpu().numpy()
            elif k == 'sinm':
                vals = summary_tensors[dir.name][category][k].cpu().numpy()
            else:
                vals = summary_tensors[dir.name][category][k][:, :, 0].cpu().numpy()
            evaluator.plot_attention_scores(dir.name, category, summary_tensors[dir.name][category], k)
            print(f"Processed summary for {dir.name}_{category}_{k} with range {(vals.min(), vals.max())}")
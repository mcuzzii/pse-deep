import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
from river.drift import ADWIN
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_stocks, get_elapsed_time, get_text_window
from collections import Counter
from experiments import Experiment, mcc_curve
from utils import setup_plot_style, COLORS
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
import time
import joblib
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.special import expit
from scipy.stats import wilcoxon
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns
import itertools
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

setup_plot_style()

def expanding_window_thresholds(val_targets, val_scores, test_targets, test_scores):
    """
    val_targets, val_scores: (N_val, S) numpy arrays
    test_targets, test_scores: (N_test, S) numpy arrays
    Returns best_thresholds: (N_test, S) numpy array
    """
    N_test, S = test_targets.shape
    best_thresholds = torch.zeros((N_test, S), dtype=torch.float32)
    chunk_size = N_test // 10 + 1

    # seed: optimize on val only
    val_mccs, thresholds = mcc_curve(val_targets, val_scores)
    current_thresholds = thresholds[torch.argmax(val_mccs, dim=0)]  # (S,)

    for i in range(0, N_test, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, N_test)

        # apply threshold from previous window to current window
        best_thresholds[start_idx:end_idx] = current_thresholds.unsqueeze(0).expand(end_idx - start_idx, -1).contiguous()

        # optimize on val + all test so far for next window
        combined_targets = torch.cat([
            torch.tensor(val_targets, device=device, dtype=torch.int32),
            torch.tensor(test_targets[:end_idx], device=device, dtype=torch.int32)
        ])
        combined_scores = torch.cat([
            torch.tensor(val_scores, device=device, dtype=torch.float32),
            torch.tensor(test_scores[:end_idx], device=device, dtype=torch.float32)
        ])
        mccs, thresholds = mcc_curve(combined_targets, combined_scores)
        current_thresholds = thresholds[torch.argmax(mccs, dim=0)]  # (S,)

    return best_thresholds.numpy()

def compute_drift(loss):

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
    
    msd = (loss - means).abs()

    msd = (msd - msd.min()) / (msd.max() - msd.min())
    mean_squared_loss_deviations = msd.mean(dim=0)

    width_histories = (width_histories - width_histories.min()) / (width_histories.max() - width_histories.min())
    drift_from_width = 1 - width_histories.mean(dim=0)

    msd_mean = msd.mean().item()
    widths_mean = 1 - width_histories.mean().item()
    combined_drift_score_mean = msd_mean * widths_mean

    return mean_squared_loss_deviations, drift_from_width, msd_mean, widths_mean, combined_drift_score_mean

def analyze(
    df,
    outcome,
    cluster_1,
    cluster_2,
    factors,
    formula,
    out_dir=None,
    two_clusters=True,
):

    if two_clusters:
        group1 = pd.factorize(df[cluster_1])[0].astype(int)
        group2 = pd.factorize(df[cluster_2])[0].astype(int)
        groups = np.column_stack((group1, group2))
    else:
        groups = pd.factorize(df[cluster_1])[0].astype(int)

    model = smf.ols(
        f"{outcome} ~ {formula}",
        data=df,
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
            "df_correction": True,
        },
    )

    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
        "ci_lower": model.conf_int()[0],
        "ci_upper": model.conf_int()[1],
        "abs_coef": model.params.abs(),
    }).sort_values("abs_coef", ascending=False)

    coef_mask = np.ones(len(coef_df), dtype=bool)
    for fe in [cluster_1, cluster_2]:
        coef_mask &= ~coef_df.index.str.contains(fe)

    coef_df = coef_df.loc[coef_mask]
    coef_df.to_csv(out_dir / f"{outcome}_coefficients.csv")

    def marginal_prediction(base_row):
        """
        Returns prediction averaged over any categorical fixed effects.
        """

        mask = np.ones(len(df), dtype=bool)

        for col, val in base_row.items():
            mask &= df[col] == val
        
        rows = df.loc[mask]

        if rows.empty:
            return np.nan

        pred = model.predict(rows)
        return pred.mean()

    configs = df[factors].drop_duplicates().to_dict(orient='index')

    marginal_rows = []

    for val in configs.values():

        marginal_rows.append({
            **val,
            f"{outcome}_pred": marginal_prediction(val),
        })

    config_df = (
        pd.DataFrame(marginal_rows)
        .sort_values(f"{outcome}_pred", ascending=False)
    )

    config_df.to_csv(
        out_dir / f"{outcome}_marginal_means.csv",
        index=False,
    )

    simple_effects = []

    for focal in factors:

        others = [f for f in factors if f != focal]

        for vals in itertools.product([0, 1], repeat=len(others)):

            cond = dict(zip(others, vals))

            row0 = {**cond, focal: 0}
            row1 = {**cond, focal: 1}

            pred0 = marginal_prediction(row0)
            pred1 = marginal_prediction(row1)

            rows0 = df.loc[
                np.logical_and.reduce(
                    [df[c] == v for c, v in row0.items()]
                )
            ]

            rows1 = df.loc[
                np.logical_and.reduce(
                    [df[c] == v for c, v in row1.items()]
                )
            ]

            design_info = model.model.data.design_info

            exog0 = build_design_matrices(
                [design_info],
                rows0,
            )[0]

            exog1 = build_design_matrices(
                [design_info],
                rows1,
            )[0]

            L = exog1.mean(axis=0) - exog0.mean(axis=0)

            test = model.t_test(L)

            simple_effects.append({
                "focal_factor": focal,
                **cond,
                "pred_at_0": pred0,
                "pred_at_1": pred1,
                "effect": float(test.effect),
                "std_err": float(test.sd),
                "t": float(test.tvalue),
                "p_value": float(test.pvalue),
                "ci_lower": float(test.conf_int()[0][0]),
                "ci_upper": float(test.conf_int()[0][1]),
            })

    simple_df = pd.DataFrame(simple_effects)

    simple_df.to_csv(
        out_dir / f"{outcome}_simple_effects.csv",
        index=False,
    )

    resids = model.resid

    if 'model' in df.columns:
        diagnose_model_dependence(
            df,
            resids,
            model_col="model",
            out_dir=out_dir,
        )

    _, p_shapiro = stats.shapiro(resids)

    pd.DataFrame({
        "residual": resids,
    }).to_csv(
        out_dir / f"{outcome}_residuals.csv",
        index=False,
    )

    pd.DataFrame([{
        "shapiro_p": p_shapiro,
        "resid_mean": resids.mean(),
        "resid_std": resids.std(),
        "resid_skew": pd.Series(resids).skew(),
        "resid_kurt": pd.Series(resids).kurt(),
    }]).to_csv(
        out_dir / f"{outcome}_residual_stats.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(resids, bins=30)
    axes[0].set_title(f"{outcome} residuals")

    stats.probplot(resids, plot=axes[1])
    axes[1].set_title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(
        out_dir / f"{outcome}_residual_diagnostics.png",
        dpi=150,
    )
    plt.close()

    tmp = df.copy()
    tmp["residuals"] = resids

    residual_wide = tmp.pivot_table(
        index=cluster_1,
        columns=cluster_2,
        values="residuals",
        aggfunc='mean'
    )

    plot_correlation_heatmap(
        pd.DataFrame(
            residual_wide.values,
            columns=residual_wide.columns,
        ),
        residual_wide.index.tolist(),
        out_dir / f"{outcome}_residual_correlation_heatmap_cluster1.png",
        f"{outcome.upper()} correlation",
    )

    plot_correlation_heatmap(
        pd.DataFrame(
            residual_wide.values.T,
            columns=residual_wide.index,
        ),
        residual_wide.columns.tolist(),
        out_dir / f"{outcome}_residual_correlation_heatmap_cluster2.png",
        f"{outcome.upper()} correlation",
    )

    return model

def valid_times(ts, offset, pred_horizon):
    last_valid_min = '00' if pred_horizon == 30 else '40'
    return (
        ts.minute.isin(range(0 + offset, 60 + offset, pred_horizon)) &
        (
            (ts.time <= pd.Timestamp(f'11:{last_valid_min}').time()) |
            (
                (ts.time >= pd.Timestamp('13:00').time()) &
                (ts.time <= pd.Timestamp(f'14:{last_valid_min}').time())
            )
        )
    )

def get_best_dataset(score, mixed_effects_path, mlp_only=True):
    marginal_means = pd.read_csv(mixed_effects_path / f'{score}_marginal_means.csv')
    best_model = marginal_means.iloc[
        np.argmax(marginal_means[f'{score}_pred'].values)
        if score != 'drift'
        else np.argmin(marginal_means[f'{score}_pred'].values)
    ]
    
    news_pre = 'news_' if best_model['news'] else ''
    social_pre = 'social_' if best_model['social'] else ''
    transformer = 'transformer_' if best_model['transformer'] and not mlp_only else 'mlp_'
    pred_hr = '30' if best_model['pred_30'] else '10'

    return f'stock_{news_pre}{social_pre}{transformer}{pred_hr}'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import statsmodels.formula.api as smf
import pandas as pd


def diagnose_model_dependence(
    df,
    residuals,
    model_col="model",
    out_dir=None,
):
    """
    Quantifies how much residual variance is attributable to model.

    Parameters
    ----------
    df : DataFrame
    residuals : array-like
        OLS residuals
    """

    tmp = df.copy()
    tmp["residual"] = residuals

    # Total residual variance
    total_var = tmp["residual"].var(ddof=1)

    # Model means
    model_means = (
        tmp
        .groupby(model_col)["residual"]
        .transform("mean")
    )

    between_var = model_means.var(ddof=1)

    within_var = (
        tmp["residual"] - model_means
    ).var(ddof=1)

    pseudo_icc = between_var / total_var

    print("\nResidual variance decomposition")
    print("--------------------------------")
    print(f"Total variance      : {total_var:.6f}")
    print(f"Between-model       : {between_var:.6f}")
    print(f"Within-model        : {within_var:.6f}")
    print(f"Pseudo ICC          : {pseudo_icc:.4f}")

    # ANOVA F-test
    anova = smf.ols(
        "residual ~ C(model)",
        data=tmp,
    ).fit()

    table = sm.stats.anova_lm(anova, typ=2)

    print("\nResidual ANOVA")
    print(table)

    if out_dir is not None:

        pd.DataFrame([{
            "total_variance": total_var,
            "between_model_variance": between_var,
            "within_model_variance": within_var,
            "pseudo_icc": pseudo_icc,
            "anova_F": table.loc["C(model)", "F"],
            "anova_p": table.loc["C(model)", "PR(>F)"],
        }]).to_csv(
            out_dir / "model_dependence_diagnostic.csv",
            index=False,
        )

    return {
        "pseudo_icc": pseudo_icc,
        "anova": table,
    }

def descriptive_stats(score_dict, metric_name, out_dir):
    rows = []
    for model_name, scores in score_dict.items():
        rows.append({
            'model':  model_name,
            'mean':   scores.mean(),
            'median': np.median(scores),
            'std':    scores.std(),
            'min':    scores.min(),
            'max':    scores.max(),
            'q25':    np.percentile(scores, 25),
            'q75':    np.percentile(scores, 75),
            'n_positive': (scores > 0).sum(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'descriptive_stats_{metric_name}.csv', index=False)
    return df

def get_price_tensor(ts, pred_horizon, offset):

    reference = pd.read_csv(
        f'experiments/results/trading_sim/close_prices/{pred_horizon}_{offset}.csv',
        index_col=0
    )
    reference.index = pd.to_datetime(reference.index)
    reference = reference.loc[reference.index.get_level_values(0).isin(ts)]
    init_prices = pd.read_csv(
        f'experiments/results/trading_sim/close_prices/init_{pred_horizon}_{offset}.csv',
        index_col=0
    )
    close_tensor = torch.tensor(reference.values, dtype=torch.float32).to(device)
    init_tensor = torch.tensor(init_prices.values, dtype=torch.float32).to(device)
    price_tensor = torch.cat([init_tensor.transpose(0, 1), close_tensor], dim=0)

    ts_mask = ts.minute.isin(range(0 + offset, 60 + offset, pred_horizon))

    return price_tensor, ts_mask, reference

def plot_correlation_heatmap(df, labels, out_path, title):
    setup_plot_style()

    if len(labels) > 1000:
        df = df.sample(1000)
        labels = df.index.tolist()

    corr = np.abs(df.T.corr().to_numpy())

    # build a viridis-like colormap from your brand colors
    viridis_cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_viridis',
        [COLORS['purple'], COLORS['indigo'], COLORS['teal'],
         COLORS['seafoam'], COLORS['green'], COLORS['yellow']]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        corr,
        xticklabels=labels if len(labels) <= 30 else False,
        yticklabels=labels if len(labels) <= 30 else False,
        cmap=viridis_cmap,
        vmin=0, vmax=1,
        center=0.5,
        square=True,
        linewidths=0,
        linecolor='white',
        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
        ax=ax
    )

    ax.set_title(title, fontsize=16, pad=16)
    ax.tick_params(axis='both', labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    return corr

def update_dict(d, key, v):
    if not (old_v := d.get(key, None)):
        d[key] = v
    else:
        for k in old_v:
            d[key][k] += v[k]

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

            logit_scores = out['test_logit_scores']
            targets = out['test_all_targets']                                                   # N, S

            if 'mcc_scores' not in out:

                overall_scores[dir.name] = dict()

                val_calib_thresholds = model['best_threshold']                                  # (S,)
                val_logit_scores = model['val_logit_scores']
                softmax_scores = torch.softmax(logit_scores, dim=-1)[..., -1]                   # N, S, 2 -> N, S
                val_softmax_scores = torch.softmax(val_logit_scores, dim=-1)[..., -1]
                val_targets = model['val_all_targets']

                best_thresholds = torch.zeros_like(targets, dtype=torch.float32)
                for i in range(0, logit_scores.shape[0], (logit_scores.shape[0] // 10 + 1)):
                    start_idx = i
                    end_idx = min(i + logit_scores.shape[0] // 10 + 1, logit_scores.shape[0])

                    if i == 0:
                        print(f'Best thresholds: {val_calib_thresholds}')
                        best_thresholds[start_idx:end_idx] = val_calib_thresholds.unsqueeze(0).expand(end_idx - start_idx, -1).contiguous()
                        print(f'Thresholds input: {val_calib_thresholds.unsqueeze(0).expand(end_idx - start_idx, -1).shape}')
                        print(f'Thresholds tensor: {best_thresholds[start_idx:end_idx]}')
                        print(f'Thresholds tensor shape: {best_thresholds[start_idx:end_idx].shape}')
                    else:
                        mccs, thresholds = mcc_curve(
                            torch.cat([val_targets, targets[:start_idx]]),
                            torch.cat([val_softmax_scores, softmax_scores[:start_idx]])
                        )                                                                       # (T, S), (T,)

                        best_idxs = torch.argmax(mccs, dim=0)
                        thresholds = thresholds[best_idxs]

                        print(f'Best thresholds: {thresholds}')
                        best_thresholds[start_idx:end_idx] = thresholds.unsqueeze(0).expand(end_idx - start_idx, -1).contiguous()
                        print(f'Thresholds tensor: {best_thresholds[start_idx:end_idx]}')
                        print(f'Thresholds tensor shape: {best_thresholds[start_idx:end_idx].shape}')

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

                torch.save(out, test_outputs)

                overall_scores[dir.name].update({
                    'test_accuracy_rolling': (preds_flat == targets_flat).float().mean().item(),
                    'test_mcc_rolling': matthews_corrcoef(targets_np, preds_np),
                    'test_precision_rolling': precision_score(targets_np, preds_np),
                    'test_recall_rolling': recall_score(targets_np, preds_np),
                    'test_f1_rolling': f1_score(targets_np, preds_np),
                    'test_precision_neg_rolling': precision_score(1 - targets_np, 1 - preds_np),
                    'test_recall_neg_rolling': recall_score(1 - targets_np, 1 - preds_np),
                    'test_f1_neg_rolling': f1_score(1 - targets_np, 1 - preds_np),
                })
            
            if 'drift_from_width' not in out:

                if dir.name not in overall_scores:
                    overall_scores[dir.name] = dict()

                criterion = nn.CrossEntropyLoss(reduction='none')
                loss = criterion(logit_scores.permute(0, 2, 1), targets)                        # N, S

                mean_squared_loss_deviations, drift_from_width, msd_mean, widths_mean, combined_drift_score_mean = compute_drift(loss)

                print(f'Drift scores: {mean_squared_loss_deviations}')
                print(f'Drift from width: {drift_from_width}')

                out['mean_squared_loss_deviations'] = mean_squared_loss_deviations
                out['drift_from_width'] = drift_from_width
                out['combined_drift_scores'] = mean_squared_loss_deviations * drift_from_width

                torch.save(out, test_outputs)

                overall_scores[dir.name].update({
                    'msd_mean': msd_mean,
                    'widths_mean': widths_mean,
                    'combined_drift_score_mean': combined_drift_score_mean
                })
        
        if not overall_scores:
            return
        
        overall_scores = pd.DataFrame.from_dict(overall_scores, orient='index')
        model_scores[overall_scores.columns] = overall_scores
        model_scores.to_csv(self.results_path / 'model_scores.csv')

    def main_and_interaction_effects(self):

        mcc_df = pd.DataFrame()
        drift_df = pd.DataFrame()

        self.stock_map = torch.load(
            self.results_path / 'reference' / 'stock_maps.pt',
            map_location=device,
            weights_only=False
        )

        out_dir = self.results_path / 'mixed_effects'
        out_dir.mkdir(exist_ok=True)

        for dir in self.experiments_path.iterdir():

            if dir.name in ('data', 'experiments', 'results'):
                continue
            
            test_outputs = dir / 'test_outputs.pt'
            out = torch.load(test_outputs, map_location=device, weights_only=False)

            mcc = out['mcc_scores']
            drift = out['combined_drift_scores']

            reorder = torch.argmax(self.stock_map[dir.name]['stock_map'], dim=-1)

            mcc_reordered = torch.zeros_like(mcc)
            drift_reordered = torch.zeros_like(drift)

            mcc_reordered[reorder] = mcc
            drift_reordered[reorder] = drift

            mcc_df[dir.name] = pd.Series(mcc_reordered.cpu().numpy())
            drift_df[dir.name] = pd.Series(drift_reordered.cpu().numpy())
        
        stock_ids = next(iter(self.stock_map.values()))['stocks']

        mcc_df['stock_id'] = stock_ids
        drift_df['stock_id'] = stock_ids

        plot_correlation_heatmap(
            mcc_df[[c for c in mcc_df.columns if c != 'stock_id']],
            [s.upper() for s in mcc_df['stock_id'].tolist()],
            self.results_path / 'mixed_effects' / 'stock_mcc_correlation_bet_models.png',
            'MCC Correlation between Stocks'
        )

        plot_correlation_heatmap(
            drift_df[[c for c in mcc_df.columns if c != 'stock_id']],
            [s.upper() for s in drift_df['stock_id'].tolist()],
            self.results_path / 'mixed_effects' / 'stock_drift_correlation_bet_models.png',
            'Drift Correlation between Stocks'
        )

        mcc_df = mcc_df.melt(id_vars=['stock_id'], var_name='setting', value_name='mcc')
        drift_df = drift_df.melt(id_vars=['stock_id'], var_name='setting', value_name='drift')

        for df in [mcc_df, drift_df]:
            df['transformer'] = df['setting'].str.contains('transformer').astype(int)
            df['news'] = df['setting'].str.contains('news').astype(int)
            df['social'] = df['setting'].str.contains('social').astype(int)
            df['pred_30'] = df['setting'].str.contains('30').astype(int)

        factors = ['transformer', 'news', 'social', 'pred_30']
        formula_two_way = f"C(stock_id) + ({' + '.join(factors)})**2"

        analyze(mcc_df, 'mcc', 'stock_id', 'setting', factors, formula_two_way, out_dir)
        analyze(drift_df, 'drift', 'stock_id', 'setting', factors, formula_two_way, out_dir)

        print(f"All results saved to {out_dir}")
    
    def _train_ml_models(self, score):

        train_path = self.experiments_path / 'data' / f'{score}m_train.pt'
        val_path = self.experiments_path / 'data' / f'{score}m_val.pt'
        test_path = self.experiments_path / 'data' / f'{score}m_test.pt'

        out_dir = self.results_path / 'baseline_models'
        out_dir.mkdir(exist_ok=True)

        score_dir = out_dir / score
        score_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading tensors ({score})...")

        train_data = torch.load(train_path, weights_only=True)
        X_train = train_data['X'].numpy()
        y_train = train_data['y'].numpy()

        val_data = torch.load(val_path, weights_only=True)
        X_val   = val_data['X'].numpy()
        y_val   = val_data['y'].numpy()

        test_data = torch.load(test_path, weights_only=True)
        X_test  = test_data['X'].numpy()
        y_test  = test_data['y'].numpy()

        N_val      = X_val.shape[0]
        N_test     = X_test.shape[0]
        S          = 30

        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # reshape to (S, N, F) and (S, N) for per-stock threshold optimization
        y_val_s   = y_val.reshape(S, N_val // S)
        y_test_s  = y_test.reshape(S, N_test // S)

        n_jobs = os.cpu_count()
        print(f"Using {n_jobs} CPU cores\n")

        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                n_jobs=n_jobs,
                verbose=2,
            ),
            'linear_svc': LinearSVC(
                max_iter=2000,
                verbose=2,
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                n_jobs=n_jobs,
                verbose=2,
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                n_jobs=n_jobs,
                device='cuda',
                verbosity=2,
                eval_metric='logloss',
            ),
        }

        model_outs = dict()
        for name, model in models.items():
            model_path = score_dir / f'{name}.joblib'

            if model_path.exists():
                print(f"Model {name}.joblib already saved. Skipping training...")
                model = joblib.load(model_path)
                train_time = 0
            else:
                print(f"\n{'='*60}")
                print(f"Training: {name.upper()}")
                print(f"{'='*60}")
                t0 = time.time()

                if name == 'xgboost':
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)
                else:
                    model.fit(X_train, y_train)

                train_time = time.time() - t0
                print(f"\nTraining time: {train_time:.1f}s")

                joblib.dump(model, model_path)
                print(f"Model saved to {model_path}")

            # --- get scores per stock ---
            print("Getting scores per stock for loss computation...")
            if name == 'linear_svc':
                # LinearSVC has no predict_proba; use decision_function
                val_scores_flat  = model.decision_function(X_val)
                test_scores_flat = model.decision_function(X_test)
                probs_pos = expit(test_scores_flat)
                probs = np.stack([1 - probs_pos, probs_pos], axis=1)
                loss = -np.log(probs[np.arange(len(y_test)).astype(int), y_test.astype(int)] + 1e-8)
            else:
                probs = model.predict_proba(X_test)  # (N, 2)
                loss = -np.log(probs[np.arange(len(y_test)).astype(int), y_test.astype(int)] + 1e-8)  # (N,) per-sample cross-entropy
                val_scores_flat  = model.predict_proba(X_val)[:, 1]
                test_scores_flat = probs[:, 1]

            # reshape to (N_per_stock, S) for threshold optimization
            val_scores_s  = val_scores_flat.reshape(S, N_val // S).T    # (N_val//S, S)
            test_scores_s = test_scores_flat.reshape(S, N_test // S).T  # (N_test//S, S)
            val_targets_s  = y_val_s.T    # (N_val//S, S)
            test_targets_s = y_test_s.T   # (N_test//S, S)
            loss_s = torch.tensor(loss.reshape(S, N_test // S).T, dtype=torch.float32)     # (N_test//S, S)

            torch.save(loss_s, score_dir / f'{name}_y_loss.pt')
            torch.save(test_scores_s, score_dir / f'{name}_probs.pt')

            model_outs[name] = (val_scores_s, test_scores_s, val_targets_s, test_targets_s, loss_s, train_time)
        
        return score_dir, S, y_test_s, model_outs

    def train_baseline_models(self):
        
        mcc_filename = get_best_dataset('mcc', self.results_path / 'mixed_effects')
        drift_filename = get_best_dataset('drift', self.results_path / 'mixed_effects')

        for score in {mcc_filename, drift_filename}:

            score_dir, S, y_test_s, model_outs = self._train_ml_models(score)
            
            if (score_dir / 'baseline_results.csv').exists():
                results = pd.read_csv(score_dir / 'baseline_results.csv', index_col=0).to_dict(orient='index')
            else:
                results = dict()
            
            for name in model_outs:
                metrics = results[name] if name in results else dict()
                val_scores_s, test_scores_s, val_targets_s, test_targets_s, loss_s, train_time = model_outs[name]

                if score == mcc_filename:

                    # --- expanding window threshold optimization ---
                    if (score_dir / f'{name}_y_preds.pt').exists():
                        print(f'Model preds already saved at {(score_dir / f'{name}_y_preds.pt')}... Skipping')
                        y_pred_per_stock = torch.load(score_dir / f'{name}_y_preds.pt', map_location=device, weights_only=True).numpy()
                        y_pred = y_pred_per_stock.flatten()
                        y_test_flat = y_test_s.flatten()
                    else:
                        print("Running expanding-window threshold optimization...")
                        t2 = time.time()
                        best_thresholds = expanding_window_thresholds(
                            val_targets_s, val_scores_s, test_targets_s, test_scores_s
                        )  # (N_test//S, S)
                        print(f"Threshold optimization time: {time.time() - t2:.1f}s")

                        # apply thresholds and flatten back
                        y_pred = (test_scores_s >= best_thresholds).astype(int).T.flatten()  # (S, N_test//S) -> flat
                        y_test_flat = y_test_s.flatten()

                        # per-stock MCC
                        y_pred_s = (test_scores_s >= best_thresholds).astype(int)  # (N_test//S, S)
                        per_stock_mcc = np.array([
                            matthews_corrcoef(y_test_s[s], y_pred_s[:, s])
                            for s in range(S)
                        ])
                        pd.DataFrame({'stock_id': range(S), 'mcc': per_stock_mcc}).to_csv(
                            score_dir / f'{name}_per_stock_mcc.csv', index=False
                        )
                        y_pred_per_stock = (test_scores_s >= best_thresholds).astype(int).T  # (S, N_test//S)
                        torch.save(
                            torch.tensor(y_pred_per_stock, dtype=torch.int32),
                            score_dir / f'{name}_y_preds.pt'
                        )
                    
                    t1 = time.time()
                    
                    metrics.update({
                        'mcc': matthews_corrcoef(y_test_flat, y_pred),
                        'accuracy': accuracy_score(y_test_flat, y_pred),
                        'precision': precision_score(y_test_flat, y_pred),
                        'recall': recall_score(y_test_flat, y_pred),
                        'f1': f1_score(y_test_flat, y_pred),
                        'precision_neg': precision_score(1 - y_test_flat, 1 - y_pred),
                        'recall_neg': recall_score(1 - y_test_flat, 1 - y_pred),
                        'f1_neg': f1_score(1 - y_test_flat, 1 - y_pred),
                        'train_time_s': train_time,
                        'pred_time_s':  time.time() - t1
                    })

                    print(f"MCC: {metrics['mcc']:.4f}")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"F1: {metrics['f1']:.4f}")
                    print(f"Negative Precision: {metrics['precision_neg']:.4f}")
                    print(f"Negative Recall: {metrics['recall_neg']:.4f}")
                    print(f"Negative F1: {metrics['f1_neg']:.4f}")
                    
                if score == drift_filename:

                    # --- computing drift scores ---
                    print("Computing drift scores")

                    mean_squared_loss_deviations, drift_from_width, msd_mean, widths_mean, combined_drift_score_mean = compute_drift(loss_s)

                    pd.DataFrame({
                        'stock_id': range(S),
                        'mean_squared_loss_deviations': mean_squared_loss_deviations.numpy(),
                        'drift_from_width': drift_from_width.numpy(),
                        'combined_drift_scores': (mean_squared_loss_deviations * drift_from_width).numpy()
                    }).to_csv(score_dir / f'{name}_per_stock_drift.csv', index=False)

                    t1 = time.time()

                    metrics.update({
                        'msd_mean': msd_mean,
                        'widths_mean': widths_mean,
                        'combined_drift_score_mean': combined_drift_score_mean,
                        'train_time_s': train_time,
                        'pred_time_s': time.time() - t1
                    })

                    print(f"MSD: {metrics['msd_mean']:.4f}")
                    print(f"Widths: {metrics['widths_mean']:.4f}")
                    print(f"Combined Drift: {metrics['combined_drift_score_mean']:.4f}")
                
                results[name] = metrics

            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df.to_csv(score_dir / 'baseline_results.csv')
            print(f"\nAll results saved to {score_dir / 'baseline_results.csv'}")
            print(results_df.to_string())

    def _dl_best_scores(self, score, mixed_effects_path):
        marginal_means = pd.read_csv(mixed_effects_path / f'{score}_marginal_means.csv')
        best_row = marginal_means.iloc[
            np.argmax(marginal_means[f'{score}_pred'].values)
            if score == 'mcc'
            else np.argmin(marginal_means[f'{score}_pred'].values)
        ]

        news_pre = 'news_' if best_row['news'] else ''
        social_pre = 'social_' if best_row['social'] else ''
        transformer = 'transformer_' if best_row['transformer'] else 'mlp_'
        pred_hr = '30' if best_row['pred_30'] else '10'

        best_dl_name = f'stock_{news_pre}{social_pre}{transformer}{pred_hr}'

        dl_test_outputs = self.experiments_path / best_dl_name / 'test_outputs.pt'
        dl_out = torch.load(dl_test_outputs, map_location=device, weights_only=False)

        dl_reorder = torch.argmax(self.stock_map[best_dl_name]['stock_map'], dim=-1)

        score_tensor = dl_out['mcc_scores' if score == 'mcc' else 'combined_drift_scores']
        score_reordered = torch.zeros_like(score_tensor)
        score_reordered[dl_reorder] = score_tensor

        print(f'Best model for {score}: {best_dl_name}')

        return score_reordered.cpu().numpy()
    
    def _ml_best_scores(self, model_name, score, mixed_effects_path):
        score_dir = get_best_dataset(score, mixed_effects_path)
        score_path = self.results_path / 'baseline_models' / score_dir / f'{model_name}_per_stock_{score}.csv'
        score_df = pd.read_csv(score_path)

        score_col = 'mcc' if score == 'mcc' else 'combined_drift_scores'
        score_values = torch.tensor(score_df[score_col].values, dtype=torch.float32, device=device)
        score_reorder = torch.argmax(self.stock_map[score_dir]['stock_map'], dim=-1)
        score_reordered = torch.zeros_like(score_values)
        score_reordered[score_reorder] = score_values

        return score_reordered.cpu().numpy()

    def main_baseline_comparison(self):

        out_dir = (self.results_path / 'baseline_comparison')
        out_dir.mkdir(exist_ok=True)

        baseline_names = ['logistic_regression', 'linear_svc', 'random_forest', 'xgboost']

        dl_mcc_per_stock   = self._dl_best_scores('mcc', self.results_path / 'mixed_effects')             # (S,)
        dl_drift_per_stock = self._dl_best_scores('drift', self.results_path / 'mixed_effects')           # (S,

        S = len(dl_mcc_per_stock)

        all_mcc = {'deep_learning': dl_mcc_per_stock}
        all_drift = {'deep_learning': dl_drift_per_stock}

        for name in baseline_names:
            all_mcc[name] = self._ml_best_scores(name, 'mcc', self.results_path / 'mixed_effects')
            all_drift[name] = self._ml_best_scores(name, 'drift', self.results_path / 'mixed_effects')
        
        mcc_summary_df = pd.DataFrame(all_mcc)
        drift_summary_df = pd.DataFrame(all_drift)

        mcc_best_model = get_best_dataset('mcc', self.results_path / 'mixed_effects', False)
        drift_best_model = get_best_dataset('drift', self.results_path / 'mixed_effects', False)

        mcc_summary_df['stock_id'] = self.stock_map[mcc_best_model]['stocks']
        drift_summary_df['stock_id'] = self.stock_map[drift_best_model]['stocks']

        mcc_summary_df = mcc_summary_df.melt(id_vars='stock_id', var_name='setting', value_name='mcc')
        drift_summary_df = drift_summary_df.melt(id_vars='stock_id', var_name='setting', value_name='drift')

        for baseline in baseline_names:
            mcc_summary_df[baseline] = (mcc_summary_df['setting'] == baseline).astype(int)
            drift_summary_df[baseline] = (drift_summary_df['setting'] == baseline).astype(int)

        mcc_summary_df.to_csv(out_dir / 'per_stock_mcc_all_models.csv', index=False)
        drift_summary_df.to_csv(out_dir / 'per_stock_drift_all_models.csv', index=False)

        formula = f"C(stock_id) + {' + '.join(baseline_names)}"

        analyze(mcc_summary_df, 'mcc', 'stock_id', 'setting', baseline_names, formula, out_dir, False)
        analyze(drift_summary_df, 'drift', 'stock_id', 'setting', baseline_names, formula, out_dir, False)

        descriptive_stats(all_mcc,   'mcc', out_dir)
        descriptive_stats(all_drift, 'drift', out_dir)

        print(f"All results saved to {out_dir}")

    def get_closing_prices(self, force=False):

        out_dir = self.results_path / 'trading_sim'
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / 'close_prices').exists() and not force:
            print("Closing prices already saved. Skipping...")
            return

        ref_30 = joblib.load('data/processed/ac_30m.joblib').filtered_date_times
        ref_10 = joblib.load('data/processed/ac_10m.joblib').filtered_date_times

        ts_30 = ref_30[int(len(ref_30) * 0.9) + 1:]
        ts_10 = ref_10[int(len(ref_10) * 0.9) + 1:]

        stocks = get_stocks()

        close_prices = {
            f'10_{offset}': pd.DataFrame(index=ts_10[ts_10.minute.isin(range(0 + offset, 60 + offset, 10))])
            for offset in range(0, 10)
        }
        close_prices.update({
            f'30_{offset}': pd.DataFrame(index=ts_30[ts_30.minute.isin(range(0 + offset, 60 + offset, 30))])
            for offset in range(0, 30)
        })
        init_prices = {
            f'10_{offset}': pd.Series()
            for offset in range(0, 10)
        }
        init_prices.update({
            f'30_{offset}': pd.Series()
            for offset in range(0, 30)
        })
        for stock in stocks:
            print(f'Computing for {stock}...')
            stock_df = DataSource()
            stock_df.create_df(stock)

            for offset in range(0, 10):
                close_prices[f'10_{offset}'][stock] = stock_df.df[f'{stock}_close'].shift(-10)[close_prices[f'10_{offset}'].index]
                init_prices[f'10_{offset}'][stock] = stock_df.df.loc[close_prices[f'10_{offset}'].index[0], f'{stock}_close']
            
            for offset in range(0, 30):
                close_prices[f'30_{offset}'][stock] = stock_df.df[f'{stock}_close'].shift(-30)[close_prices[f'30_{offset}'].index]
                init_prices[f'30_{offset}'][stock] = stock_df.df.loc[close_prices[f'30_{offset}'].index[0], f'{stock}_close']
        
        close_prices_dir = out_dir / 'close_prices'
        close_prices_dir.mkdir(parents=True, exist_ok=True)

        for key, value in close_prices.items():
            value.to_csv(close_prices_dir / f'{key}.csv')
        for key, value in init_prices.items():
            value.to_csv(close_prices_dir / f'init_{key}.csv')
    
    def _compute_profits(self, probs, price_tensor, stock_map, k, offset, model_name):
        top_k = torch.topk(probs, k + 1).indices                                 # n, k
        bottom_k = torch.topk(probs, k + 1, largest=False).indices               # n, k

        long_before = torch.gather(price_tensor[:-1], 1, stock_map[top_k])                  # n, k
        long_after = torch.gather(price_tensor[1:], 1, stock_map[top_k])                    # n, k

        short_before = torch.gather(price_tensor[:-1], 1, stock_map[bottom_k])              # n, k
        short_after = torch.gather(price_tensor[1:], 1, stock_map[bottom_k])                # n, k

        long_profits = (long_after / long_before).sum(dim=-1)
        short_profits = (short_before / short_after).sum(dim=-1)

        mean_profits = (long_profits + short_profits) / (2 * (k + 1))

        profits = torch.cumprod(mean_profits, dim=0)

        plt.figure(figsize=(8, 5))

        plt.plot(profits.cpu().numpy(), label='Profits')

        plt.xlabel("Time")
        plt.ylabel("Money")
        plt.title(f"Trading Simulation")
        plt.grid(False)
        plt.tight_layout()

        plots_path = self.results_path / 'trading_sim' / 'plots'
        plots_path.mkdir(parents=True, exist_ok=True)

        save_path = plots_path / f'{model_name}_{k + 1}_{offset}.png'

        plt.savefig(save_path, dpi=300)
        plt.close()

        return profits
    
    def trading_simulations(self, force=False):

        if (self.results_path / 'trading_sim' / 'results.pt').exists() and not force:
            print("Trading simulations already implemented, skipping...")
            return

        ref_30 = joblib.load('data/processed/ac_30m.joblib')
        ref_10 = joblib.load('data/processed/ac_10m.joblib')

        ts_30 = ref_30.filtered_date_times
        ts_10 = ref_10.filtered_date_times

        ts_30 = ts_30[int(len(ts_30) * 0.9) + 1:]
        ts_10 = ts_10[int(len(ts_10) * 0.9) + 1:]

        c_30 = ref_30.features.index('ac_close')
        c_10 = ref_10.features.index('ac_close')

        results_dict = dict()
        ref_dict = dict()

        for dir in self.experiments_path.iterdir():

            if dir.name in ('data', 'experiments', 'results'):
                continue
                
            print(f'Simulating for {dir.name}...')

            out_path = dir / 'test_outputs.pt'
            out = torch.load(out_path, map_location=device, weights_only=False)

            logits = out['test_logit_scores']                           # N, 30, 2
            softmax_scores = torch.softmax(logits, dim=-1)[..., -1]     # N, 30

            news = 'news' in dir.name
            social = 'social' in dir.name
            transformer = 'transformer' in dir.name
            pred_30 = '30' in dir.name

            pred_horizon = 30 if pred_30 else 10
            c_idx = c_30 if pred_30 else c_10
            ts = ts_30 if pred_30 else ts_10

            if news and not transformer:
                news_df = joblib.load(f'data/processed/news_{pred_horizon}m.joblib')
                ts = ts.intersection(news_df.df.dropna().index)

            data_fn = (
                f'stock_transformer_{pred_horizon}m_test.pt'
                if transformer else (
                    f'stock_{'news_' if news else ''}{'social_' if social else ''}mlp_{pred_horizon}m_test.pt'
                )
            )

            features = torch.load(
                self.experiments_path / 'data' / data_fn,
                map_location=device,
                weights_only=True
            )['features' if transformer else 'X']                                       # 30, N, features if transformer; N*30, features if mlp

            if transformer:
                close = features[:, -len(ts):, c_idx]                                # 30, N current close prices
            else:
                features = features.reshape(30, features.shape[0] // 30, -1)            # 30, N, features
                close = features[:, :, c_idx]                                           # 30, N current close prices
            
            close = torch.roll(close, -pred_horizon, 1)                                 # 30, N future close prices

            results_dict[dir.name] = dict()

            for offset in range(pred_horizon):

                filtered_close = close[:, valid_times(ts, offset, pred_horizon)]        # 30, n filtered future close prices
                
                price_tensor, ts_mask, reference = get_price_tensor(ts, pred_horizon, offset)

                filtered_ref = reference.loc[valid_times(reference.index, offset, pred_horizon)].values     # 30, n filtered future close prices
                filtered_ref = torch.tensor(filtered_ref, dtype=torch.float32).transpose(0, 1).to(device)

                corr_matrix = torch.corrcoef(torch.cat([filtered_ref, filtered_close], dim=0))              # 60, 60
                stock_map = torch.argmax(corr_matrix[-30:, :30], dim=-1)

                if offset == 0:
                    ref_dict[dir.name] = {
                        'stocks': reference.columns.tolist(),
                        'stock_map': corr_matrix[-30:, :30]
                    }

                results_dict[dir.name][offset] = dict()

                for k in range(15):
                    
                    profits = self._compute_profits(
                        softmax_scores[ts_mask],
                        price_tensor,
                        stock_map, k,
                        offset, dir.name
                    )

                    results_dict[dir.name][offset][k] = profits

        torch.save(results_dict, self.results_path / 'trading_sim' / 'results.pt')

        ref_dir = self.results_path / 'reference'
        ref_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ref_dict, ref_dir / 'stock_maps.pt')
    
    def interpret_trading_sim(self, score=None):

        results = torch.load(
            self.results_path / 'trading_sim' / 'results.pt',
            map_location=device,
            weights_only=False
        )
        if score is not None:
            best_model_name = get_best_dataset(
                'cum_profit',
                self.results_path / 'trading_sim' / 'mixed_effects',
                False
            )
            best_model = results[best_model_name]
            results = torch.load(
                self.results_path / 'trading_sim' / 'baseline_results.pt',
                map_location=device,
                weights_only=False
            )
            results[best_model_name] = best_model

        ref_30 = joblib.load('data/processed/ac_30m.joblib')
        ref_10 = joblib.load('data/processed/ac_10m.joblib')

        ts_30 = ref_30.filtered_date_times
        ts_10 = ref_10.filtered_date_times

        ts_30 = ts_30[int(len(ts_30) * 0.9) + 1:]
        ts_10 = ts_10[int(len(ts_10) * 0.9) + 1:]

        ts_all = ts_30.union(ts_10)
        
        summary_df = pd.DataFrame(index=ts_all.floor('10min').unique().union(ts_all.floor('30min').unique()))
        snapshots_dir = self.results_path / 'trading_sim' / 'snapshots'
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        final_returns_per_model = dict()
        for key, value in results.items():
            if score is None or key.startswith('stock'):
                model_name = key
            else:
                model_name = score

            news = 'news' in model_name
            social = 'social' in model_name
            transformer = 'transformer' in model_name
            pred_30 = '30' in model_name

            pred_horizon = 30 if pred_30 else 10
            ts = ts_30 if pred_30 else ts_10

            if news and not transformer:
                news_df = joblib.load(f'data/processed/news_{pred_horizon}m.joblib')
                ts = ts.intersection(news_df.df.dropna().index)
            
            model_df = pd.DataFrame(index=ts_all)

            final_returns_per_offset = []
            num_offset = len(value.items())
            for i, (offset_key, offset) in enumerate(value.items()):
                reference = pd.read_csv(
                    f'experiments/results/trading_sim/close_prices/{pred_horizon}_{offset_key}.csv',
                    index_col=0
                )
                reference.index = pd.to_datetime(reference.index)
                reference = reference.loc[reference.index.get_level_values(0).isin(ts)]

                offset_tensor = torch.stack(list(offset.values()), dim=0)               # k (N,) -> (k, N)
                if i % (num_offset // 10) == 0:
                    final_returns_per_offset.append(offset_tensor[:, -1])               # k

                offset_df = pd.DataFrame(offset_tensor.cpu().numpy().T, index=reference.index)
                offset_df = offset_df.add_suffix(f'_{offset_key}')
                model_df = model_df.join(offset_df, how='left')
            
            final_returns_per_model[key] = torch.cat(final_returns_per_offset, dim=0).cpu().numpy()   # k * offset
            
            model_df = model_df.reset_index().melt(id_vars='local_time').dropna()
            group_freq = '30min' if pred_30 else '10min'
            model_df = model_df.loc[
                (model_df['local_time'].dt.time != pd.Timestamp('11:30' if pred_30 else '11:50').time()) &
                (model_df['local_time'].dt.time != pd.Timestamp('14:30' if pred_30 else '14:50').time())
            ]
            groups = model_df['local_time'].dt.floor(group_freq)
            group_means = model_df.groupby(groups)['value'].mean()
            summary_df.loc[group_means.index, key] = group_means.values

            model_df.to_csv(snapshots_dir / f'model_{key}.csv')
            summary_df.to_csv(snapshots_dir / f'summary_{key}.csv')
        
        summary_df = summary_df.reset_index().melt(id_vars='local_time', var_name='setting', value_name='profit_perc')

        if score is None:
            summary_df['transformer'] = np.where(
                summary_df['setting'].str.contains('transformer'),
                'Transformer',
                'MLP'
            )
            summary_df['news'] = np.where(
                summary_df['setting'].str.contains('news').astype(int),
                'With news',
                'No news'
            )
            summary_df['social'] = np.where(
                summary_df['setting'].str.contains('social').astype(int),
                'With social media',
                'Without social media'
            )
            summary_df['pred_30'] = np.where(
                summary_df['setting'].str.contains('30').astype(int),
                '30-min return target',
                '10-min return target'
            )
            summary_df.drop(columns=['setting'], inplace=True)

            summary_df['time_idx'] = summary_df.groupby(['transformer', 'pred_30', 'news', 'social']).cumcount()

            palette = {'With news': COLORS['purple'], 'No news': COLORS['green']}
            dashes = {'With social media': (1, 0), 'Without social media': (4, 1.5)}

            col_order = ['Transformer', 'MLP']
            row_order = ['10-min return target', '30-min return target']

            g = sns.FacetGrid(
                summary_df,
                row='pred_30', col='transformer',
                row_order=row_order, col_order=col_order,
                height=3.5, aspect=1.6,
                despine=True,
            )

            g.map_dataframe(
                sns.lineplot, x='time_idx', y='profit_perc',
                hue='news', style='social',
                palette=palette, dashes=dashes,
            )

            g.set_axis_labels('Time', 'Cumulative Return')
            g.set_titles(row_template='{row_name}', col_template='{col_name}')
            g.add_legend(title='')
            g.figure.subplots_adjust(top=0.92)
            g.figure.suptitle('Trading Simulation Results', fontsize=13, fontweight='bold')

            legend = g.legend
            for text in legend.get_texts():
                if text.get_text() in ('news', 'social'):
                    text.set_visible(False)

            for ax in g.axes.flat:
                ax.axhline(1.0, color='black', linewidth=0.6, linestyle=':', alpha=0.5)
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))

            g.savefig(self.results_path / 'trading_sim' / 'overall.png', dpi=300, bbox_inches='tight')

            tsim_df = pd.DataFrame(final_returns_per_model)
            tsim_df['k_offset_pair_id'] = range(len(tsim_df))
            tsim_df = tsim_df.melt(id_vars='k_offset_pair_id', var_name='setting', value_name='cum_profit')

            tsim_df['transformer'] = tsim_df['setting'].str.contains('transformer').astype(int)
            tsim_df['news'] = tsim_df['setting'].str.contains('news').astype(int)
            tsim_df['social'] = tsim_df['setting'].str.contains('social').astype(int)
            tsim_df['pred_30'] = tsim_df['setting'].str.contains('30').astype(int)

            factors = ['transformer', 'news', 'social', 'pred_30']
            formula_two_way = f"C(k_offset_pair_id) + ({' + '.join(factors)})**2"
            out_dir = self.results_path / 'trading_sim' / 'mixed_effects'
            out_dir.mkdir(exist_ok=True)

            analyze(tsim_df, 'cum_profit', 'k_offset_pair_id', 'setting', factors, formula_two_way, out_dir)

            print(f"All results saved to {out_dir}")

        else:
            summary_df['setting'] = np.where(
                summary_df['setting'] == 'logistic_regression',
                'Logistic Regression',
                np.where(
                    summary_df['setting'] == 'linear_svc',
                    'Linear SVC',
                    np.where(
                        summary_df['setting'] == 'random_forest',
                        'Random Forest',
                        np.where(
                            summary_df['setting'] == 'xgboost',
                            'XGBoost',
                            'Best DL model'
                        )
                    )
                )
            )
            summary_df['time_idx'] = summary_df.groupby('setting').cumcount()

            palette = {
                'Logistic Regression': COLORS['green'],
                'Linear SVC': COLORS['yellow'],
                'Random Forest': COLORS['indigo'],
                'XGBoost': COLORS['seafoam'],
                'Best DL model': COLORS['purple']
            }

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.lineplot(
                data=summary_df, x='time_idx', y='profit_perc',
                hue='setting', palette=palette, ax=ax
            )

            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Return')
            ax.set_title('Trading Simulation Results', fontsize=13, fontweight='bold')
            ax.axhline(1.0, color='black', linewidth=0.6, linestyle=':', alpha=0.5)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))

            legend = ax.get_legend()
            legend.set_title('')
            for text in legend.get_texts():
                if text.get_text() == 'setting':
                    text.set_visible(False)

            fig.savefig(self.results_path / 'trading_sim' / 'baseline.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            tsim_df = pd.DataFrame(final_returns_per_model)
            tsim_df['k_offset_pair_id'] = range(len(tsim_df))
            tsim_df = tsim_df.melt(id_vars='k_offset_pair_id', var_name='setting', value_name='cum_profit')

            tsim_df['logistic_regression'] = (tsim_df['setting'] == 'logistic_regression').astype(int)
            tsim_df['linear_svc'] = (tsim_df['setting'] == 'linear_svc').astype(int)
            tsim_df['random_forest'] = (tsim_df['setting'] == 'random_forest').astype(int)
            tsim_df['xgboost'] = (tsim_df['setting'] == 'xgboost').astype(int)

            factors = ['logistic_regression', 'linear_svc', 'random_forest', 'xgboost']
            formula_two_way = f"C(k_offset_pair_id) + {' + '.join(factors)}"
            out_dir = self.results_path / 'trading_sim' / 'baseline_comparison'
            out_dir.mkdir(exist_ok=True)

            analyze(tsim_df, 'cum_profit', 'k_offset_pair_id', 'setting', factors, formula_two_way, out_dir, False)

            print(f"All results saved to {out_dir}")
            descriptive_stats(final_returns_per_model, 'cum_profit', out_dir)
    
    def baseline_models_trading_sim(self):

        score = get_best_dataset('cum_profit', self.results_path / 'trading_sim' / 'mixed_effects')
        self._train_ml_models(score)

        ml_models = ['logistic_regression', 'linear_svc', 'random_forest', 'xgboost']
        baseline_models = self.results_path / 'baseline_models' / score

        pred_horizon = 30 if '30' in score else 10
        ts = joblib.load(f'data/processed/ac_{pred_horizon}m.joblib').filtered_date_times
        ts = ts[int(0.9 * len(ts)) + 1:]

        if 'news' in score:
            news_df = joblib.load(f'data/processed/news_{pred_horizon}m.joblib')
            ts = ts.intersection(news_df.df.dropna().index)

        results_dict = dict()

        for model in ml_models:
            probs = torch.load(baseline_models / f'{model}_probs.pt', map_location=device, weights_only=False)       # N, S

            results_dict[model] = dict()
            for offset in range(pred_horizon):

                price_tensor, ts_mask, _ = get_price_tensor(ts, pred_horizon, offset)

                results_dict[model][offset] = dict()

                for k in range(15):
                    profits = self._compute_profits(
                        torch.tensor(probs[ts_mask], dtype=torch.float32, device=device),
                        price_tensor,
                        torch.argmax(self.stock_map[score]['stock_map'], dim=-1),
                        k, offset, model
                    )

                    results_dict[model][offset][k] = profits
        
        torch.save(results_dict, self.results_path / 'trading_sim' / 'baseline_results.pt')
    
    def interpret_baseline_models_trading_sim(self):

        score = get_best_dataset('cum_profit', self.results_path / 'trading_sim' / 'mixed_effects')
        self.interpret_trading_sim(score)
    
    def interpret_shap_values(self):

        ref_30 = joblib.load('data/processed/ac_30m.joblib').filtered_date_times
        ref_10 = joblib.load('data/processed/ac_10m.joblib').filtered_date_times

        ref_30 = ref_30[int(len(ref_30) * 0.9) + 1:]
        ref_10 = ref_10[int(len(ref_10) * 0.9) + 1:]

        shap_dfs = dict()

        for dir in self.experiments_path.iterdir():

            pred_30 = '30' in dir.name
            news = 'news' in dir.name
            social = 'social' in dir.name
            transformer = 'transformer' in dir.name
            pred_horizon = 30 if pred_30 else 10

            if news and not transformer:
                news_30 = joblib.load(f'data/processed/news_30m.joblib')
                news_10 = joblib.load(f'data/processed/news_10m.joblib')
                ts_30 = ref_30.intersection(news_30.df.dropna().index)
                ts_10 = ref_10.intersection(news_10.df.dropna().index)
            else:
                ts_30 = ref_30.copy()
                ts_10 = ref_10.copy()
            
            ts = ts_30 if pred_30 else ts_10

            if dir.name in ('data', 'results'):
                continue

            out = torch.load(
                dir / 'test_outputs.pt',
                map_location=device,
                weights_only=False
            )

            sv = out['test_shap_values'].to(torch.float32)

            if transformer:                                                     # sv: M, g, 30
                test_y = torch.load(
                    self.experiments_path / 'data' / f'stock_transformer_{pred_horizon}m_test.pt',
                    map_location=device,
                    weights_only=False
                )['target']

                y_id = torch.arange(test_y.shape[2], device=device)
                mask = (y_id % 88.0) < 2

                ts = ts[mask.cpu().numpy()]

                model_prefix = 'tfm'

            else:                                                               # sv: M, g, 1
                test_y = torch.load(
                    self.experiments_path / 'data' / f'{dir.name}m_test.pt',
                    map_location=device,
                    weights_only=False
                )['y']                                                     # N

                y_id = torch.arange(len(test_y), device=device)

                mask = (y_id % 64.0) < 32
                if mask[-1]:
                    mask = mask & (y_id < len(test_y) - 32)

                reshuffled_sv = torch.full((test_y.shape[0], sv.shape[1]), float('nan'), device=device)
                reshuffled_sv[mask] = sv.squeeze(-1)
                reshuffled_sv = reshuffled_sv.reshape(30, reshuffled_sv.shape[0] // 30, -1)
                sv = reshuffled_sv.permute(1, 2, 0)

                model_prefix = 'mlp'
            
            elapsed_time = get_elapsed_time(ts, min(ts_30.min(), ts_10.min()))
            time_of_day = np.where(
                ts.time <= pd.Timestamp('10:30').time(),
                'market_open',
                np.where(
                    ts.time <= pd.Timestamp('12:00').time(),
                    'pre_recess',
                    np.where(
                        ts.time <= pd.Timestamp('13:45').time(),
                        'pm_open',
                        'pre_close'
                    )
                )
            )

            stock_labs = self.stock_map[dir.name]['stocks']
            stock_map = self.stock_map[dir.name]['stock_map']

            stock_reorder = torch.argmax(stock_map, dim=-1)

            sv_reordered = torch.zeros_like(sv)
            sv_reordered[:, :, stock_reorder] = sv
            sv = sv_reordered

            grps = out['shap_group_names']

            for key in grps:
                idx = grps.index(key)
                df = pd.DataFrame(sv[:, idx, :].cpu().numpy(), columns=stock_labs)

                df['timestamp'] = ts
                df['elapsed_time'] = elapsed_time
                df['time_of_day'] = time_of_day
                df['explainer_timestamp'] = ts.astype(str) + f' - {dir.name}'
                df['model'] = dir.name

                df = df.melt(
                    id_vars=['timestamp', 'elapsed_time', 'time_of_day', 'explainer_timestamp', 'model'],
                    var_name='stock',
                    value_name='shap'
                )

                df['stock_timestamp'] = df['timestamp'].astype(str) + f' - ' + df['stock']

                df['pred_30'] = pred_30
                df['news'] = news
                df['social'] = social

                df = df.dropna()

                shap_name = f'{model_prefix}_{key}'
                shap_dfs[shap_name] = pd.concat([shap_dfs[shap_name], df]) if shap_name in shap_dfs else df
                shap_dfs[shap_name] = shap_dfs[shap_name].reset_index(drop=True)

        out_dir = self.results_path / 'shap_analysis'
        out_dir.mkdir(parents=True, exist_ok=True)

        df_dir = out_dir / 'model_inputs'
        df_dir.mkdir(parents=True, exist_ok=True)

        for key, df in shap_dfs.items():
            df = df[df.nunique()[df.nunique() > 1].index]

            groups = df.groupby(['time_of_day', 'model', 'stock'])['shap'].count()
            groups.to_csv(df_dir / f'{key}_groups.csv')

            df.to_csv(df_dir / f'{key}.csv', index=False)

            factors = [c for c in df.columns if c in ('news', 'social', 'pred_30')]
            factor_terms = " + ".join(factors)
            formula = (
                f"C(stock) + "
                f"elapsed_time + "
                f"C(time_of_day) + "
                f"{factor_terms} + "
                f"elapsed_time:({factor_terms}) + "
                f"C(time_of_day):({factor_terms}) + "
                f"({factor_terms})**2"
            )

            res_dir = out_dir / key
            res_dir.mkdir(parents=True, exist_ok=True)

            analyze(df, 'shap', 'stock', 'explainer_timestamp', factors, formula, res_dir)
    
    def get_embeddings(self):

        for mode in ('news', 'social'):
            for dir in self.experiments_path.iterdir():
                if dir.name in ('data', 'results') or 'mlp' in dir.name or mode not in dir.name:
                    continue

                news = 'news' in dir.name
                social = 'social' in dir.name
                transformer = True
                pred_30 = '30' in dir.name
                pred_horizon_prefix = 30 if pred_30 else 10

                experiment = Experiment(
                    experiment_name=dir.name,
                    transformer=transformer,
                    pred_30=pred_30,
                    news=news,
                    social=social,
                    stock_lookback=60
                )
                experiment.build_model(
                    input_dim=100 if transformer else 110,
                    news_input_dim=15,
                    social_input_dim=6 if transformer else 15,
                    text_input_dim=1024,
                    social_embedding_dim=16,
                    hidden_dim=384,
                    embedding_dim=128,
                    num_layers=1 if transformer else 5,
                    temporal_embedding_dim=16,
                    dropout=0.1,
                    K=5,
                    num_samples=500,
                    sigma=5e-2,
                )

                best_path = self.experiments_path / dir.name / f'{dir.name}.pt'
                best_weights = torch.load(best_path, map_location=device, weights_only=False)['model']

                model = experiment.model.to(device)
                model.load_state_dict(best_weights)
                model.eval()

                embed_model = model.news_embed if mode == 'news' else model.social_embed

                data_path = f'experiments/data/{mode}_transformer_{pred_horizon_prefix}m_test.pt'
                data_tensor = torch.load(data_path, map_location=device)

                timestamps = data_tensor['timestamps'].unsqueeze(0)
                embeddings = data_tensor['embeddings'].unsqueeze(0)
                
                if mode == 'social':
                    impact = data_tensor['impact'].unsqueeze(0)
                
                args = (embeddings, timestamps) if mode == 'news' else (embeddings, impact, timestamps)

                out = embed_model(*args).squeeze(0)

                print(out.shape)

                out_dir = self.results_path / 'attn_analysis' / 'embeds'
                out_dir.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'out': out,
                    'timestamps': timestamps.squeeze(0)
                }, out_dir / f'{dir.name}_{mode}.pt')
    
    def interpret_attention_scores(self):

        print("Loading reference datasets..")

        ts_30 = joblib.load('data/processed/ac_30m.joblib').filtered_date_times
        ts_10 = joblib.load('data/processed/ac_10m.joblib').filtered_date_times

        print("Loading text datasets..")

        news_df = joblib.load('data/processed/news.joblib')
        social_df = joblib.load('data/processed/social_media.joblib')

        summary_tensors = dict()

        for dir in self.experiments_path.iterdir():
            if dir.name in ('data', 'results') or 'mlp' in dir.name:
                continue

            batches_dir = dir / 'weights'

            pred_30 = '30' in dir.name
            news = 'news' in dir.name
            social = 'social' in dir.name
            pred_horizon = 30 if pred_30 else 10

            ts = ts_30 if pred_30 else ts_10
            ts = ts[int(len(ts) * 0.9) + 1:]

            if news:
                news_data = torch.load(
                    self.results_path / 'attn_analysis' / 'embeds' / f'{dir.name}_news.pt',
                    map_location=device,
                    weights_only=False
                )
                news_embeds = news_data['out']
                news_ts = news_data['timestamps']
            
            if social:
                social_data = torch.load(
                    self.results_path / 'attn_analysis' / 'embeds' / f'{dir.name}_social.pt',
                    map_location=device,
                    weights_only=False
                )
                social_embeds = social_data['out']
                social_ts = social_data['timestamps']

            summary_tensors[dir.name] = dict()

            counter = 0
            
            for batch in tqdm(batches_dir.iterdir()):
                counter += 1
                if counter == 30: break
                tensors = torch.load(batch, map_location=device, weights_only=False)

                i = int(batch.name[6:-3]) * 2 + 1

                if news and social:
                    keys = ('tst', 'sft', 'nft', 'ist', 'sin', 'nin')
                elif social:
                    keys = ('tst', 'sft', 'ist', 'sin')
                elif news:
                    keys = ('tst', 'nft', 'ist' ,'nin')
                else:
                    keys = ('tst', 'ist')
                
                snapshot = {k: v for k, v in zip(keys, tensors)}

                if social or news:
        
                    last_timestamp = get_elapsed_time(ts[i]).astype(np.float32)
                    time_vec_input = get_elapsed_time(ts).astype(np.float32)
                    idx = (pd.Series(time_vec_input) - last_timestamp).abs().idxmin()
        
                    cutoff, _ = get_text_window(idx, time_vec_input.index, pred_horizon)

                    cutoff_scaled = get_elapsed_time(cutoff)
                
                ignore_batch = False

                for ind in ('sin', 'nin'):
                    if ind not in snapshot:
                        continue

                    if ind == 'sin':
                        text_ts = social_ts
                        text_embeds = social_embeds
                        text_df = social_df
                        attn = 'sft'
                    else:
                        text_ts = news_ts
                        text_embeds = news_embeds
                        text_df = news_df
                        attn = 'nft'

                    mask = (cutoff_scaled < text_ts) & (text_ts <= last_timestamp)
                    sample = text_embeds[mask]        # Tn, En

                    try:

                        selected = torch.einsum("stkn,ne->stke", snapshot[ind], sample)      # S, Ts, K, En

                        selected_norm = F.normalize(selected, dim=-1)      # S, Ts, K, En
                        sample_norm   = F.normalize(sample, dim=-1)        # Tn, En

                        sim = torch.einsum("stke,ne->stkn", selected_norm, sample_norm)  # S, Ts, K, Tn

                        closest_idx = sim.argmax(dim=-1)   # S, Ts, K

                        Tn = sample.shape[0]

                        scores = torch.zeros(Tn, device=selected.device, dtype=snapshot[attn].dtype)
                        scores = scores.scatter_add_(0, closest_idx.reshape(-1), snapshot[attn].reshape(-1))

                        text = text_df.df.loc[
                            (cutoff < text_df.df.index) & (text_df.df.index <= ts[i]),
                            text_df.text_col
                        ]

                        snapshot[ind] = Counter(dict(zip(text, scores.tolist())))
                    
                    except Exception as e:
                        print(f"Exception raised: {e}")
                        ignore_batch = True
                        break
                    
                if ignore_batch:
                    continue

                for ind in ('tst', 'sft', 'nft', 'ist'):
                    if ind not in snapshot:
                        continue

                    snapshot[ind] = snapshot[ind].sum(dim=0)
                
                if ts[i].time() <= pd.Timestamp('10:00').time():
                    update_dict(summary_tensors[dir.name], 'market_open', snapshot)
                elif ts[i].time() <= pd.Timestamp('12:00').time():
                    update_dict(summary_tensors[dir.name], 'am_session', snapshot)
                elif ts[i].time() <= pd.Timestamp('14:15').time():
                    update_dict(summary_tensors[dir.name], 'pm_session', snapshot)
                else:
                    update_dict(summary_tensors[dir.name], 'market_close', snapshot)
                
                update_dict(summary_tensors[dir.name], str(ts[i].date()), snapshot)
                update_dict(summary_tensors[dir.name], 'overall', snapshot)
            
        torch.save(summary_tensors, self.results_path / 'attn_analysis' / 'summary_tensors.pt')
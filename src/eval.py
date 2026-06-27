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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_stocks
from experiments import mcc_curve
import statsmodels.formula.api as smf
import os
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.special import expit
from scipy.stats import wilcoxon
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    msd = ((loss - means) ** 2)

    msd = (msd - msd.min()) / (msd.max() - msd.min())
    mean_squared_loss_deviations = msd.mean(dim=0)

    width_histories = (width_histories - width_histories.min()) / (width_histories.max() - width_histories.min())
    drift_from_width = 1 - width_histories.mean(dim=0)

    msd_mean = msd.mean().item()
    widths_mean = 1 - width_histories.mean().item()
    combined_drift_score_mean = msd_mean * widths_mean

    return mean_squared_loss_deviations, drift_from_width, msd_mean, widths_mean, combined_drift_score_mean

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

    def random_intercept_mixed_effects(self):
        import itertools
        from scipy import stats
        import matplotlib.pyplot as plt

        mcc_df = pd.DataFrame()
        drift_df = pd.DataFrame()
        for dir in self.experiments_path.iterdir():

            if dir.name in ('data', 'experiments', 'results'):
                continue
            
            test_outputs = dir / 'test_outputs.pt'
            out = torch.load(test_outputs, map_location=device, weights_only=False)

            mcc_df[dir.name] = pd.Series(out['mcc_scores'].cpu().numpy())
            drift_df[dir.name] = pd.Series(out['drift_from_width'].cpu().numpy())
        
        mcc_df['stock_id'] = range(len(mcc_df))
        drift_df['stock_id'] = range(len(drift_df))

        mcc_df = mcc_df.melt(id_vars=['stock_id'], var_name='setting', value_name='mcc')
        drift_df = drift_df.melt(id_vars=['stock_id'], var_name='setting', value_name='drift')

        for df in [mcc_df, drift_df]:
            df['transformer'] = df['setting'].str.contains('transformer').astype(int)
            df['news'] = df['setting'].str.contains('news').astype(int)
            df['social'] = df['setting'].str.contains('social').astype(int)
            df['pred_30'] = df['setting'].str.contains('30').astype(int)
            df.drop(columns=['setting'], inplace=True)

        factors = ['transformer', 'news', 'social', 'pred_30']
        formula_two_way = "(transformer + news + social + pred_30)**2"
        formula_main = "transformer + news + social + pred_30"
        out_dir = self.results_path / 'mixed_effects'
        out_dir.mkdir(exist_ok=True)

        def analyze(df, outcome):
            # --- Fit models ---
            model_2way = smf.mixedlm(f"{outcome} ~ {formula_two_way}", data=df, groups=df["stock_id"]).fit(reml=False)
            model_main = smf.mixedlm(f"{outcome} ~ {formula_main}", data=df, groups=df["stock_id"]).fit(reml=False)
            model_reml = smf.mixedlm(f"{outcome} ~ {formula_two_way}", data=df, groups=df["stock_id"]).fit()

            # --- Coefficients ---
            coef_df = pd.DataFrame({
                'coef': model_reml.fe_params,
                'std_err': model_reml.bse_fe,
                'z': model_reml.tvalues,
                'p_value': model_reml.pvalues,
                'ci_lower': model_reml.conf_int()[0],
                'ci_upper': model_reml.conf_int()[1],
                'abs_coef': model_reml.fe_params.abs()
            }).sort_values('abs_coef', ascending=False)
            coef_df.to_csv(out_dir / f'{outcome}_coefficients.csv')

            # --- Model comparison ---
            lr_stat = 2 * (model_2way.llf - model_main.llf)
            df_diff = len(model_2way.params) - len(model_main.params)
            p_lrt = stats.chi2.sf(lr_stat, df_diff)
            group_var = model_reml.cov_re.iloc[0, 0]
            resid_var = model_reml.scale
            icc = group_var / (group_var + resid_var)
            model_comparison_df = pd.DataFrame([{
                'lr_stat': lr_stat,
                'df_diff': df_diff,
                'p_lrt': p_lrt,
                'aic_two_way': model_2way.aic,
                'aic_main': model_main.aic,
                'bic_two_way': model_2way.bic,
                'bic_main': model_main.bic,
                'icc': icc,
                'group_var': group_var,
                'resid_var': resid_var,
            }])
            model_comparison_df.to_csv(out_dir / f'{outcome}_model_comparison.csv', index=False)

            # --- Marginal means ---
            configs = list(itertools.product([0, 1], repeat=4))
            config_df = pd.DataFrame(configs, columns=factors)
            config_df['stock_id'] = 0
            config_df[f'{outcome}_pred'] = model_reml.predict(config_df)
            config_df = config_df.drop(columns=['stock_id']).sort_values(f'{outcome}_pred', ascending=False)
            config_df.to_csv(out_dir / f'{outcome}_marginal_means.csv', index=False)

            # --- Simple effects ---
            simple_effects_rows = []
            for focal in factors:
                others = [f for f in factors if f != focal]
                for vals in itertools.product([0, 1], repeat=len(others)):
                    cond = dict(zip(others, vals))
                    row0 = {**cond, focal: 0, 'stock_id': 0}
                    row1 = {**cond, focal: 1, 'stock_id': 0}
                    pred0 = model_reml.predict(pd.DataFrame([row0]))[0]
                    pred1 = model_reml.predict(pd.DataFrame([row1]))[0]
                    simple_effects_rows.append({
                        'focal_factor': focal,
                        **cond,
                        'effect': pred1 - pred0,
                        'pred_at_0': pred0,
                        'pred_at_1': pred1,
                    })
            simple_effects_df = pd.DataFrame(simple_effects_rows)
            simple_effects_df.to_csv(out_dir / f'{outcome}_simple_effects.csv', index=False)

            # --- Residual diagnostics ---
            resids = model_reml.resid
            _, p_shapiro = stats.shapiro(resids)
            resid_df = pd.DataFrame({
                'residual': resids.values,
            })
            resid_df.to_csv(out_dir / f'{outcome}_residuals.csv', index=False)
            pd.DataFrame([{
                'shapiro_p': p_shapiro,
                'resid_mean': resids.mean(),
                'resid_std': resids.std(),
                'resid_skew': resids.skew(),
                'resid_kurt': resids.kurt(),
            }]).to_csv(out_dir / f'{outcome}_residual_stats.csv', index=False)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(resids, bins=30)
            axes[0].set_title(f'{outcome} residuals histogram')
            stats.probplot(resids, plot=axes[1])
            axes[1].set_title(f'{outcome} Q-Q plot')
            plt.tight_layout()
            plt.savefig(out_dir / f'{outcome}_residual_diagnostics.png', dpi=150)
            plt.close()

            return model_reml

        analyze(mcc_df, 'mcc')
        analyze(drift_df, 'drift')

        print(f"All results saved to {out_dir}")

    def train_baseline_models(self):

        marginal_means = pd.read_csv(self.results_path / 'mixed_effects' / 'mcc_marginal_means.csv')
        best_model = marginal_means.iloc[np.argmax(marginal_means['mcc_pred'].values)]
        
        news_pre = 'news_' if best_model['news'] else ''
        social_pre = 'social_' if best_model['social'] else ''
        pred_hr = '30m_' if best_model['pred_30'] else '10m_'

        data_filename = f'stock_{news_pre}{social_pre}mlp_{pred_hr}'

        train_path = self.experiments_path / 'data' / f'{data_filename}train.pt'
        val_path = self.experiments_path / 'data' / f'{data_filename}val.pt'
        test_path = self.experiments_path / 'data' / f'{data_filename}test.pt'

        out_dir = self.results_path / 'baseline_models'
        out_dir.mkdir(exist_ok=True)

        print(f"Loading tensors ({data_filename})...")

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

        results = {}
        for name, model in models.items():
            model_path = out_dir / f'{name}.joblib'

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

            torch.save(loss_s, out_dir / f'{name}_y_loss.pt')

            # --- computing drift scores ---
            print("Computing drift scores")

            mean_squared_loss_deviations, drift_from_width, msd_mean, widths_mean, combined_drift_score_mean = compute_drift(loss_s)

            pd.DataFrame({
                'stock_id': range(S),
                'mean_squared_loss_deviations': mean_squared_loss_deviations.numpy(),
                'drift_from_width': drift_from_width.numpy(),
                'combined_drift_scores': (mean_squared_loss_deviations * drift_from_width).numpy()
            }).to_csv(out_dir / f'{name}_per_stock_drift.csv', index=False)

            metrics = {
                'msd_mean': msd_mean,
                'widths_mean': widths_mean,
                'combined_drift_score_mean': combined_drift_score_mean
            }

            # --- expanding window threshold optimization ---
            if (out_dir / f'{name}_y_preds.pt').exists():
                print(f'Model preds already saved at {(out_dir / f'{name}_y_preds.pt')}... Skipping')
                y_pred_per_stock = torch.load(out_dir / f'{name}_y_preds.pt', map_location=device, weights_only=True).numpy()
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
                    out_dir / f'{name}_per_stock_mcc.csv', index=False
                )
                y_pred_per_stock = (test_scores_s >= best_thresholds).astype(int).T  # (S, N_test//S)
                torch.save(
                    torch.tensor(y_pred_per_stock, dtype=torch.int32),
                    out_dir / f'{name}_y_preds.pt'
                )

            t1 = time.time()
            metrics.update({
                'mcc':          matthews_corrcoef(y_test_flat, y_pred),
                'accuracy':     accuracy_score(y_test_flat, y_pred),
                'precision':    precision_score(y_test_flat, y_pred),
                'recall':       recall_score(y_test_flat, y_pred),
                'f1':           f1_score(y_test_flat, y_pred),
                'train_time_s': train_time,
                'pred_time_s':  time.time() - t1,
            })
            results[name] = metrics

            print(f"MCC:       {metrics['mcc']:.4f}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1:        {metrics['f1']:.4f}")

        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(out_dir / 'baseline_results.csv')
        print(f"\nAll results saved to {out_dir / 'baseline_results.csv'}")
        print(results_df.to_string())
        return results_df

    def wilcoxon_baseline_comparison(self):

        out_dir = self.results_path / 'baseline_comparison'
        out_dir.mkdir(exist_ok=True)

        baseline_dir = self.results_path / 'baseline_models'
        baseline_names = ['logistic_regression', 'linear_svc', 'random_forest', 'xgboost']

        # --- load deep learning model mcc and drift scores ---
        # find the best deep learning model (same logic as train_baseline_models)
        marginal_means = pd.read_csv(self.results_path / 'mixed_effects' / 'mcc_marginal_means.csv')
        best_model_row = marginal_means.iloc[np.argmax(marginal_means['mcc_pred'].values)]

        news_pre    = 'news_'        if best_model_row['news']        else ''
        social_pre  = 'social_'      if best_model_row['social']      else ''
        transformer = 'transformer_' if best_model_row['transformer'] else 'mlp_'
        pred_hr     = '30'         if best_model_row['pred_30']     else '10'

        best_dl_name = f'stock_{news_pre}{social_pre}{transformer}{pred_hr}'
        data_filename = f'stock_{news_pre}{social_pre}mlp_{pred_hr}'

        dl_test_outputs = self.experiments_path / best_dl_name / 'test_outputs.pt'
        dl_out = torch.load(dl_test_outputs, map_location=device, weights_only=False)

        dl_mcc_per_stock   = dl_out['mcc_scores'].cpu().numpy()             # (S,)
        dl_drift_per_stock = dl_out['drift_from_width'].cpu().numpy()       # (S,)

        S = len(dl_mcc_per_stock)

        # --- collect all model scores ---
        all_mcc   = {'deep_learning': dl_mcc_per_stock}
        all_drift = {'deep_learning': dl_drift_per_stock}

        for name in baseline_names:
            mcc_path = baseline_dir / f'{name}_per_stock_mcc.csv'
            mcc_df   = pd.read_csv(mcc_path)
            all_mcc[name] = mcc_df['mcc'].values  # (S,)

            drift_path = baseline_dir / f'{name}_per_stock_drift.csv'
            drift_df = pd.read_csv(drift_path)
            all_drift[name] = drift_df['drift_from_width'].values

        # --- summary dataframes ---
        mcc_summary_df   = pd.DataFrame(all_mcc,   index=[f'stock_{i}' for i in range(S)])
        drift_summary_df = pd.DataFrame(all_drift, index=[f'stock_{i}' for i in range(S)])
        mcc_summary_df.to_csv(out_dir / 'per_stock_mcc_all_models.csv')
        drift_summary_df.to_csv(out_dir / 'per_stock_drift_all_models.csv')

        # --- wilcoxon: deep learning vs each baseline ---
        all_models = ['deep_learning'] + baseline_names

        def run_wilcoxon_table(score_dict, metric_name, higher_is_better=True):
            rows = []
            dl_scores = score_dict['deep_learning']
            for name in baseline_names:
                baseline_scores = score_dict[name]
                diff = dl_scores - baseline_scores
                if np.all(diff == 0):
                    stat, p = np.nan, np.nan
                else:
                    stat, p = wilcoxon(dl_scores, baseline_scores, alternative='greater' if higher_is_better else 'less')
                mean_dl   = dl_scores.mean()
                mean_base = baseline_scores.mean()
                rows.append({
                    'baseline':        name,
                    f'mean_dl_{metric_name}':       mean_dl,
                    f'mean_baseline_{metric_name}': mean_base,
                    f'mean_diff_{metric_name}':     mean_dl - mean_base,
                    'wilcoxon_stat':   stat,
                    'p_value':         p,
                    'significant_p05': p < 0.05 if not np.isnan(p) else False,
                    'significant_p01': p < 0.01 if not np.isnan(p) else False,
                })
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / f'wilcoxon_{metric_name}.csv', index=False)
            return df

        mcc_wilcoxon_df   = run_wilcoxon_table(all_mcc,   'mcc',   higher_is_better=True)
        drift_wilcoxon_df = run_wilcoxon_table(all_drift, 'drift', higher_is_better=False)

        # --- descriptive stats per model ---
        def descriptive_stats(score_dict, metric_name):
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

        descriptive_stats(all_mcc,   'mcc')
        descriptive_stats(all_drift, 'drift')

        print(f"All results saved to {out_dir}")

    def get_closing_prices(self):

        out_dir = self.results_path / 'trading_sim'
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / 'close_prices').exists():
            print("Closing prices already saved. Skipping...")
            return

        self._ref_30 = joblib.load('data/processed/ac_30m.joblib').filtered_date_times
        self._ref_10 = joblib.load('data/processed/ac_10m.joblib').filtered_date_times

        ts_30 = self._ref_30[int(len(self._ref_30) * 0.9) + 1:]
        ts_10 = self._ref_10[int(len(self._ref_10) * 0.9) + 1:]

        stocks = get_stocks()

        close_prices = {
            f'10_{offset}': pd.DataFrame(index=ts_10[ts_10.minute.isin(range(0 + offset, 60 + offset, 10))])
            for offset in range(0, 10)
        }
        close_prices.update({
            f'30_{offset}': pd.DataFrame(index=ts_30[ts_30.minute.isin(range(0 + offset, 60 + offset, 30))])
            for offset in range(0, 30)
        })
        for stock in stocks:
            print(f'Computing for {stock}...')
            stock_df = DataSource()
            stock_df.create_df(stock)

            for offset in range(0, 10):
                close_prices[f'10_{offset}'][stock] = stock_df.df[f'{stock}_close'].shift(-10)[close_prices[f'10_{offset}'].index]
            
            for offset in range(0, 30):
                close_prices[f'30_{offset}'][stock] = stock_df.df[f'{stock}_close'].shift(-30)[close_prices[f'30_{offset}'].index]
        
        close_prices_dir = out_dir / 'close_prices'
        close_prices_dir.mkdir(parents=True, exist_ok=True)

        for key, value in close_prices.items():
            value.to_csv(close_prices_dir / f'{key}.csv')
    
    def trading_simulations(self):

        ref_30 = joblib.load('data/processed/ac_30m.joblib').filtered_date_times
        ref_10 = joblib.load('data/processed/ac_10m.joblib').filtered_date_times

        ts_30 = ref_30[int(len(ref_30) * 0.9) + 1:]
        ts_10 = ref_10[int(len(ref_10) * 0.9) + 1:]

        for dir in self.experiments_path:

            if dir.name in ('data', 'experiments', 'results'):
                continue

            out_path = dir / 'test_outputs.pt'
            out = torch.load(out_path, map_location=device, weights_only=False)

            logits = out['test_logit_scores']
# main.py

import sys
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from utils import seed_everything

seed_everything(42)

from processing import DataSource, get_unique_instruments, get_stocks
from experiments import Experiment
from eval import Eval
from dotenv import load_dotenv
import gc

load_dotenv()

def preprocess():

    print("Creating social media data...")
    social_media_data = DataSource()
    social_media_data.create_df(raw_folder_name='social', file_name='social_media', medium='x_posts', text_col='text', date_col='createdAt')
    social_media_data.get_social_sentiment_examples()

    print("Creating news data...")
    lseg_news_data = DataSource()
    lseg_news_data.create_df(raw_folder_name='news', file_name='news', medium='lseg_news')
    lseg_news_data.get_translated_examples()
    lseg_news_data.get_headline_sentiment_examples()

    for i in [52320, 47506, 32768]:
        social_media_data.get_similar_embeddings(index=i, n_results=10)
        lseg_news_data.get_similar_embeddings(index=i, n_results=10)
    
    del social_media_data, lseg_news_data
    gc.collect()

    print("Creating stock data...")
    stocks = get_unique_instruments('data/raw/stock')
    for stock in stocks:
        stock_data = DataSource()
        stock_data.create_df(raw_folder_name='stock', file_name=stock, medium='stock')
        del stock_data
        gc.collect()
    
    print("Creating government bond data...")
    bonds = get_unique_instruments('data/raw/bond')
    for bond in bonds:
        bond_data = DataSource()
        bond_data.create_df(raw_folder_name='bond', file_name=bond, medium='bond')
        del bond_data
        gc.collect()
    
    print("Creating copper price data...")
    copper = DataSource()
    copper.create_df(raw_folder_name='copper', file_name='copper', medium='copper')
    del copper
    gc.collect()

    print("Creating oil price data...")
    oil = DataSource()
    oil.create_df(raw_folder_name='crude', file_name='lcoc1', medium='oil')
    del oil
    gc.collect()

    print("Creating USD price data...")
    usd = DataSource()
    usd.create_df(raw_folder_name='forex', file_name='usd', medium='fx')
    del usd
    gc.collect()

    print("Creating XAU price data...")
    xau = DataSource()
    xau.create_df(raw_folder_name='xau', file_name='xau', medium='fx')
    del xau
    gc.collect()

    print("Combining government bonds...")
    bond_master = DataSource()
    bond_master.create_df(file_name='bond_master', medium='bonds')
    del bond_master
    gc.collect()

    print("Combining financial instruments...")
    stocks = get_stocks()
    
    for stock in stocks:
        print(f"Combining instruments for {stock}...")
        stock_data = DataSource()
        stock_data.create_df(file_name=stock, medium='combined')
        del stock_data
        gc.collect()
    
    print("Selecting features...")
    features_30 = DataSource()
    features_30.create_df(file_name='features_30m', medium='features', target=30, stocks=stocks, ignore_history=True)
    features_30.save_selected_features()
    del features_30
    gc.collect()

    features_10 = DataSource()
    features_10.create_df(file_name='features_10m', medium='features', target=10, stocks=stocks, ignore_history=True)
    features_10.save_selected_features()
    del features_10
    gc.collect()

    print("Finalizing datasets...")
    for stock in stocks:
        stock_data = DataSource()
        stock_data.create_df(file_name=stock, medium='final', target=30)
        del stock_data
        gc.collect()
    
    for stock in stocks:
        stock_data = DataSource()
        stock_data.create_df(file_name=stock, medium='final', target=10)
        del stock_data
        gc.collect()
    
    lseg_news_data = DataSource()
    lseg_news_data.create_df(file_name='news', medium='final_text')
    del lseg_news_data
    gc.collect()

    social_media_data = DataSource()
    social_media_data.create_df(file_name='social_media', medium='final_text')
    del social_media_data
    gc.collect()

    social_indicators_30 = DataSource()
    social_indicators_30.create_df('social_media', medium='social_indicators', target=30)
    del social_indicators_30
    gc.collect()

    social_indicators_10 = DataSource()
    social_indicators_10.create_df('social_media', medium='social_indicators', target=10)
    del social_indicators_10
    gc.collect()

    news_indicators_30 = DataSource()
    news_indicators_30.create_df('news', medium='news_sentiment', target=30)
    del news_indicators_30
    gc.collect()

    news_indicators_10 = DataSource()
    news_indicators_10.create_df('news', medium='news_sentiment', target=10)
    del news_indicators_10
    gc.collect()

def run_experiments():

    for transformer in (True, False):
        for social in (False, True):
            for news in (False, True):
                for pred_30 in (False, True):

                    news_prefix = 'news_' if news else ''
                    social_prefix = 'social_' if social else ''
                    model_prefix = 'transformer_' if transformer else 'mlp_'
                    pred_horizon_prefix = 30 if pred_30 else 10

                    experiment = Experiment(
                        experiment_name=f"stock_{news_prefix}{social_prefix}{model_prefix}{pred_horizon_prefix}",
                        transformer=transformer,
                        pred_30=pred_30,
                        news=news,
                        social=social,
                        stock_lookback=60
                    )
                    experiment.build_dataset()
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
                    experiment.train(
                        num_epochs=2,
                        batch_size=2 if transformer else 32,
                        accumulation_steps=16 if transformer else 1,
                        lr=1e-5 if transformer else 1e-5 / 16,
                        val_every=lambda x: (8 * x) ** 2,
                        patience=1000,
                        sigma_end=1e-5
                    )
                    experiment.plot_loss_curves()
                    experiment.threshold_optimize()
                    experiment.run_testing(force=True, recompute_only=True)

def main():
    evaluator = Eval()
    evaluator.overall_metrics()
    evaluator.compute_experiment_data()
    evaluator.random_intercept_mixed_effects()
    evaluator.train_baseline_models()
    evaluator.wilcoxon_baseline_comparison()

if __name__ == '__main__':
    main()
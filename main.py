import sys
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_unique_instruments
from dotenv import load_dotenv
import gc

load_dotenv()

def main():

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
    stocks = list(set(stocks) - {'psei', 'psho', 'psse', 'psmo', 'psfi', 'pspr', 'psin'})
    
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
        stock_data.create_df(file_name=stock, medium='final', target=30, ignore_history=True)
        del stock_data
        gc.collect()
    
    for stock in stocks:
        stock_data = DataSource()
        stock_data.create_df(file_name=stock, medium='final', target=10, ignore_history=True)
        del stock_data
        gc.collect()
    
    lseg_news_data = DataSource()
    lseg_news_data.create_df(file_name='news', medium='final_text', ignore_history=True)
    del lseg_news_data
    gc.collect()

    social_media_data = DataSource()
    social_media_data.create_df(file_name='social_media', medium='final_text', ignore_history=True)
    del social_media_data
    gc.collect()


if __name__ == '__main__':
    main()

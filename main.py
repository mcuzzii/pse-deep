import sys
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_unique_instruments
from dotenv import load_dotenv
import gc

load_dotenv()

def main():

    social_media_data = DataSource()
    social_media_data.create_df(raw_folder_name='social', file_name='social_media', medium='x_posts', text_col='text', date_col='createdAt')
    social_media_data.get_social_sentiment_examples()

    lseg_news_data = DataSource()
    lseg_news_data.create_df(raw_folder_name='news', file_name='news', medium='lseg_news')
    lseg_news_data.get_translated_examples()
    lseg_news_data.get_headline_sentiment_examples()

    for i in [52320, 47506, 32768]:
        social_media_data.get_similar_embeddings(index=i, n_results=10)
        lseg_news_data.get_similar_embeddings(index=i, n_results=10)
    
    del social_media_data, lseg_news_data
    gc.collect()
    
    stocks = get_unique_instruments('data/raw/stock')
    for stock in stocks:
        stock_data = DataSource()
        stock_data.create_df(raw_folder_name='stock', file_name=stock, medium='stock')
        del stock_data
        gc.collect()
    
    bonds = get_unique_instruments('data/raw/bond')
    for bond in bonds:
        bond_data = DataSource()
        bond_data.create_df(raw_folder_name='bond', file_name=bond, medium='bond')
        del bond_data
        gc.collect()
    
    copper = DataSource()
    copper.create_df(raw_folder_name='copper', file_name='copper', medium='copper')
    del copper
    gc.collect()

    oil = DataSource()
    oil.create_df(raw_folder_name='crude', file_name='lcoc1', medium='oil')
    del oil
    gc.collect()

    usd = DataSource()
    usd.create_df(raw_folder_name='forex', file_name='usd', medium='fx')
    del usd
    gc.collect()

    xau = DataSource()
    xau.create_df(raw_folder_name='xau', file_name='xau', medium='fx')
    del xau
    gc.collect()

    bond_master = DataSource()
    bond_master.create_df(file_name='bond_master', medium='bonds')
    del bond_master
    gc.collect()

    stocks = list(set(stocks) - {'psei', 'psho', 'psse', 'psmo', 'psfi', 'pspr', 'psin'})
    
    for stock in stocks:
        stock_data = DataSource()
        stock_data.create_df(file_name=stock, medium='combined', ignore_history=True)
        del stock_data
        gc.collect()



if __name__ == '__main__':
    main()

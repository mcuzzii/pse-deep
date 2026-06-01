import sys
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_unique_instruments
from dotenv import load_dotenv

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
    
    stocks = get_unique_instruments('data/raw/stock')
    stock_dfs = dict()
    for stock in stocks:
        stock_data = DataSource()
        stock_data.create_df(raw_folder_name='stock', file_name=stock, medium='stock')
        stock_dfs[stock_data.file_name] = stock_data
    
    bonds = get_unique_instruments('data/raw/bond')
    bond_dfs = dict()
    for bond in bonds:
        bond_data = DataSource()
        bond_data.create_df(raw_folder_name='bond', file_name=bond, medium='bond')
        bond_dfs[bond_data.file_name] = bond_data
    
    copper = DataSource()
    copper.create_df(raw_folder_name='copper', file_name='copper', medium='copper')

    oil = DataSource()
    oil.create_df(raw_folder_name='crude', file_name='lcoc1', medium='oil')

    usd = DataSource()
    usd.create_df(raw_folder_name='forex', file_name='usd', medium='fx')

    xau = DataSource()
    xau.create_df(raw_folder_name='xau', file_name='xau', medium='fx')

    bond_master = DataSource()
    bond_master.create_df(file_name='bond_master', bonds=bond_dfs, ignore_history=True)
    
    for fn in stock_dfs:
        stock_dfs[fn].combine_data(stock_dfs, copper, oil, usd, xau, ignore_history=True)



if __name__ == '__main__':
    main()

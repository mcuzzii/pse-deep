import sys
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_unique_instruments
from dotenv import load_dotenv

load_dotenv()

def main():

    social_media_data = DataSource()
    social_media_data.create_df(
        raw_path='data/raw/social',
        processed_path='data/processed',
        file_name='social_media',
        medium='x_posts',
        text_col='text',
        date_col='createdAt'
    )
    social_media_data.get_social_sentiment_examples()

    lseg_news_data = DataSource()
    lseg_news_data.create_df(
        raw_path='data/raw/news/all_news.xlsx',
        processed_path='data/processed',
        file_name='news',
        medium='lseg_news'
    )
    lseg_news_data.get_translated_examples()
    lseg_news_data.get_headline_sentiment_examples()

    for i in [52320, 47506, 32768]:
        social_media_data.get_similar_embeddings(index=i, n_results=10)
        lseg_news_data.get_similar_embeddings(index=i, n_results=10)
    
    stocks = get_unique_instruments('data/raw/stock')
    stock_dfs = []
    for stock in stocks:
        stock_data = DataSource()
        stock_data.create_df(
            raw_path=Path('data/raw/stock') / f'{stock}.xlsx',
            processed_path='data/processed',
            file_name=stock,
            medium='stock'
        )
        stock_dfs.append(stock_data.df)

if __name__ == '__main__':
    main()

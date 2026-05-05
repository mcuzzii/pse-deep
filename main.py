import sys
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource
from dotenv import load_dotenv

load_dotenv()

def main():

    social_media_data = DataSource()
    social_media_data.create_text_df(
        raw_path='data/raw/social',
        processed_path='data/processed',
        file_name='social_media',
        medium='x_posts',
        text_col='text',
        date_col='createdAt'
    )

    lseg_news_data = DataSource()
    lseg_news_data.create_text_df(
        raw_path='data/raw/news/all_news.xlsx',
        processed_path='data/processed',
        file_name='news',
        medium='lseg_news'
    )

    for i in [52320, 47506, 32768]:
        social_media_data.get_similar_embeddings(index=i, n_results=10)
        lseg_news_data.get_similar_embeddings(index=i, n_results=10)
    
    lseg_news_data.batch_process_headlines().to_csv('data/processed/lseg_news_translated.csv')

if __name__ == '__main__':
    main()

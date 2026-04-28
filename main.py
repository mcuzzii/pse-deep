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
    print(social_media_data.df.head())

    lseg_news_data = DataSource()
    lseg_news_data.create_text_df(
        raw_path='data/raw/news/all_news.xlsx',
        processed_path='data/processed',
        file_name='news',
        medium='lseg_news'
    )
    print(lseg_news_data.df.head())

if __name__ == '__main__':
    main()

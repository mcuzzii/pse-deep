import sys
from pathlib import Path

# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource
from dotenv import load_dotenv

load_dotenv()

def main():

    data = DataSource()
    data.create_social_media_df(
        raw_path='data/raw/social',
        processed_path='data/processed',
        file_name='social_media',
        text_col='text',
        date_col='createdAt',
        ignore_history=True
    )
    print(data.df.head())

if __name__ == '__main__':
    main()

import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

news = DataSource()
news.create_df('news')
news.df.sort_index().to_csv('data/samples/news_samples.csv')
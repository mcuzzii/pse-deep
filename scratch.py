import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

news_indicators_30 = DataSource()
news_indicators_30.create_df('news_30m')
news_indicators_30.df.to_csv('data/samples/news_30m.csv')
import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

news_indicators_30 = DataSource()
news_indicators_30.create_df('news', medium='news_sentiment', target=30)
del news_indicators_30
gc.collect()
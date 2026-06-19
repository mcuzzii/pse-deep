import sys
from pathlib import Path
import joblib
import gc

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

social_indicators_30 = DataSource()
social_indicators_30.create_df('social_media', medium='social_indicators', target=30, ignore_history=True)
del social_indicators_30
gc.collect()

social_indicators_10 = DataSource()
social_indicators_10.create_df('social_media', medium='social_indicators', target=10, ignore_history=True)
del social_indicators_10
gc.collect()

news_indicators_30 = DataSource()
news_indicators_30.create_df('news', medium='news_sentiment', target=30)
del news_indicators_30
gc.collect()

news_indicators_10 = DataSource()
news_indicators_10.create_df('news', medium='news_sentiment', target=10, ignore_history=True)
del news_indicators_10
gc.collect()

import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

social_indicators_10 = DataSource()
social_indicators_10.create_df('social_media', medium='social_indicators', target=10)
del social_indicators_10
gc.collect()
import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

social_final = DataSource()
social_final.create_df('social_media_30m')
social_final.df.to_csv('data/samples/social_media_30m_sample.csv')
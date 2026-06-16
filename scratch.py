import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

social_final = DataSource()
social_final.create_df('social_media', medium='social_indicators', target=30)
social_final.df.head(2000).to_csv('data/samples/social_media_30m.csv')
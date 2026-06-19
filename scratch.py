import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

social_final = DataSource()
social_final.create_df('social_media')
social_final._text_indicators()
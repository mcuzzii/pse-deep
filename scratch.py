import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

social_final = DataSource()
social_final.create_df('social_media')
social_final.df = social_final.df.set_index('created_at')
social_final._finalized_text()
joblib.dump(social_final, 'social_media.joblib')
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

social_final = DataSource()
social_final.create_df('social_media')
print(social_final.df.set_index('created_at').index)
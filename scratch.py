import joblib
import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

news_df = DataSource('news_copy')
print(news_df.df)
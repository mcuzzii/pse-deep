import sys
from pathlib import Path
import joblib

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource

news = DataSource()
news.create_df('news')
print(news.index.days.value_counts().value_counts())
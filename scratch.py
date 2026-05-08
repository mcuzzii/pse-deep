import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
import joblib
from processing import DataSource

news_data = joblib.load('data/processed/news.joblib')
print(news_data.text_col)
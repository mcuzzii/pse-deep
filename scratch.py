import joblib
import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))

news_data = joblib.load('data/processed/news.joblib').df

print(news_data.head(5))
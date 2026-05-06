import joblib
import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import DataSource

df = joblib.load('data/processed/news.joblib')

print(df.df.columns)
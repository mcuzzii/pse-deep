import sys
from pathlib import Path
# Add the 'src' directory to the path
sys.path.append(str(Path.cwd() / 'src'))
from processing import get_unique_tickers

print(get_unique_tickers('data/raw/stock'))
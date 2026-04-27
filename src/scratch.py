from processing import DataSource
from pathlib import Path

data = DataSource()
data.load_json_folder(Path('../data/raw/social'), ignore_history=True)

print(data._history)
print(data.df.head(50)['text'])
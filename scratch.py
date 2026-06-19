import sys
from pathlib import Path
import joblib
import gc

sys.path.append(str(Path.cwd() / 'src'))

from processing import DataSource, get_stocks
from experiments import Experiment

test_experiment = Experiment('text_experiment')
test_experiment.test()
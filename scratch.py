import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.append(str(Path.cwd() / 'src'))

from processing import get_stocks, DataSource

data = torch.load('experiments/data/stock_transformer_30m_test.pt', map_location=torch.device('cpu'), weights_only=True)['features']

_ref_30 = joblib.load('data/processed/ac_30m.joblib').filtered_date_times
_ref_10 = joblib.load('data/processed/ac_10m.joblib').filtered_date_times

ts_30 = _ref_30[int(len(_ref_30) * 0.9) + 1:]
ts_10 = _ref_10[int(len(_ref_10) * 0.9) + 1:]

data = np.roll(data[:, -len(ts_30):, :].numpy(), -30, 1)
print(data.shape)

for offset in range(30):

    reference = pd.read_csv(f'experiments/results/trading_sim/close_prices/30_{offset}.csv', index_col=0)
    reference.index = pd.to_datetime(reference.index)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Closing Prices')
    ax1.plot(
        data[
            1,
            (
                ts_30.minute.isin(range(0 + offset, 60 + offset, 30)) &
                (
                    (ts_30.time <= pd.Timestamp('11:00').time()) |
                    (
                        (ts_30.time >= pd.Timestamp('13:00').time()) &
                        (ts_30.time <= pd.Timestamp('14:00').time())
                    )
                )
            ),
            41
        ], label="Standardized Closing Prices")
    ax2.plot(
        reference.loc[
            (reference.index.time <= pd.Timestamp('11:00').time()) |
            (
                (reference.index.time >= pd.Timestamp('13:00').time()) &
                (reference.index.time <= pd.Timestamp('14:00').time())
            ),
            'emi'
        ].to_numpy(), label="Actual Closing Prices"
    )

    ax1.grid(False)
    ax2.grid(False)
    fig.tight_layout()

    save_path = f'experiments/results/trading_sim/emi_{offset}.png'

    fig.savefig(save_path, dpi=300)

# 0: acen
# 1: emi
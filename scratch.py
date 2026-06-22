import torch
import pandas as pd
import matplotlib.pyplot as plt

test_tensor = torch.load('experiments/data/stock_mlp_10m_test.pt', map_location='cpu', weights_only=False)

trading_day_mins = pd.Series(test_tensor['X'][:test_tensor['X'].shape[0] // 30, -1].numpy())

plt.figure(figsize=(8, 5))

plt.plot(trading_day_mins, label="Trading Day Minutes")

plt.xlabel("Time")
plt.ylabel("Minutes")
plt.title("Trading Day Minutes")
plt.grid(False)
plt.tight_layout()

save_path = 'experiments/results/data_alignment_testing.png'

plt.savefig(save_path, dpi=300)
plt.close()
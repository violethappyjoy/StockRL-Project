import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_params as params

accuracy = pd.read_csv('graphs/accuracy.csv')
loss = pd.read_csv('graphs/loss.csv')
avg_reward = pd.read_csv('graphs/reward_avg.csv')

x = accuracy.Step.to_numpy()
y = accuracy.Value.to_numpy()

plt.plot(x, y, c='b')
params = {'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2, 'font.size': 14}
plt.rcParams.update(params)
plt.xlabel('Steps')
plt.ylabel('Value')

plt.savefig('graphs/Accuracy.png', dpi = 400)
plt.savefig('graphs/Accuracy.pdf', dpi = 400)
plt.show()

x = loss.Step.to_numpy()
y = loss.Value.to_numpy()
print(y)

plt.plot(x, y, c='b')
params = {'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2, 'font.size': 14}
plt.rcParams.update(params)
plt.xlabel('Steps')
plt.ylabel('Value')

y_min = min(y)
y_max = max(y)
y_range = y_max - y_min
plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range) 

plt.savefig('graphs/Loss.png', dpi = 400)
plt.savefig('graphs/Loss.pdf', dpi = 400)
plt.show()

x = avg_reward.Step.to_numpy()
y = avg_reward.Value.to_numpy()

plt.plot(x, y, c='b')
params = {'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2, 'font.size': 14}
plt.rcParams.update(params)
plt.xlabel('Steps')
plt.ylabel('Value')

plt.savefig('graphs/Avg_Reward.png', dpi = 400)
plt.savefig('graphs/Avg_Reward.pdf', dpi = 400)

plt.show()
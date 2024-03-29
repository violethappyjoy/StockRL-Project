import numpy as np
from enum import Enum
from collections import deque
from environment.env_var import Box, OneD
import matplotlib.pyplot as plt
from matplotlib import rc_params as params
import matplotlib.colors as mcolors
import time

class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2
    
class MarketEnv:
    def __init__(self, data, window_size = 10):
        self.data = data
        self.window_size = window_size
        self.prices = self.data.loc[:, 'Close'].to_numpy()
        
        self.shape = (window_size, 1)
        self.action_space = OneD(len(Actions))
        self.observation_space = Box(low = -np.inf, high = np.inf, size = self.shape, dtype = np.float64)
        
        self._start = self.window_size
        self._end = len(self.prices) - 1
        self._prev = deque(maxlen=window_size)
        
    def reset(self):
        self._done = False
        self._current = self._start
        self._prev.extend(self.prices[self._current - self.window_size: self._current][::1])
        self._total_reward = 0
        self._total_profit = 0
        self.hist = {}
        return self._get_state()
    
    def _get_state(self):
        return self.prices[(self._current - self.window_size+1):self._current+1]
    
    def _update_hist(self, info):
        if not self.hist:
            self.hist = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.hist[key].append(value)
        
    def _update_profit(self, action):
        current_price = self.prices[self._current]
        if action == Actions.Buy.value:
            minPrice = min(self._prev)
            profit = minPrice - current_price
            # self._total_profit += profit
            return profit
        elif action == Actions.Sell.value:
            maxPrice = max(self._prev)
            profit = current_price - maxPrice
            return profit
            # self._total_profit += profit
        else:
            return 0
        
    def render(self, actions, show_colorbar=False):
        siz = len(actions)
        prices = self.prices[(self._current-siz):self._current]
        # print(f'len={len(prices)}, len action = {siz}, arr = {prices}')
        
        plt.plot(range(siz), prices, c='b', label = 'Main', zorder=1)
        action_map = mcolors.ListedColormap(['green', 'red'])
        plt.scatter(range(siz), prices, c=actions, cmap=action_map, marker='o', label = 'Actions', s=2, zorder = 2)
        params = {'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2, 'font.size': 14}
        plt.rcParams.update(params)
        
        plt.xlabel('Time(Days)')
        plt.ylabel('Prices')
        
        if show_colorbar:
            cbar = plt.colorbar(ticks=[0.50, 1.50])
            cbar.ax.set_yticklabels(['Sell', 'Buy'], rotation=270)
            cbar.set_label('Action', rotation=270)
        # plt.show()
        plt.savefig(f'graphs/action_{self._current}_{time.time()}.png', dpi=400)
        plt.savefig(f'graphs/action_{self._current}_{time.time()}.pdf', dpi=400)
            
          
            
    
    def step(self, action):
        self._prev.append(self.prices[self._current])
        self._current += 1
        
        # print(self._prev)
        # print(self._current)
        
        if self._current == self._end:
            self._done = True
        
        # print(profit)
        step_reward = self._update_profit(action)
        self._total_profit += step_reward
        # print(self._total_profit)
        
        # step_reward = np.clip(step_reward, -1, 1)
        # print(step_reward)
            
        self._total_reward += step_reward
        step = self._get_state()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit
        )
        self._update_hist(info)
        
        return step, step_reward, self._done, info
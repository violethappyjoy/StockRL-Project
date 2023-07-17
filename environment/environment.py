import numpy as np
from enum import Enum
from collections import deque
from environment.env_var import Box, OneD

class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2
    
class Position(Enum):
    Short = 0
    Long = 1
    
    def opposite(self):
        return Position.Short if self == Position.Long else Position.Long

class MarketEnv:
    # initialize values for our Stock Market Environment
    # Data(DF), window size(amt of past data)
    def __init__(self, data, window_size = 10, init_amount = 100):
        self.data = data
        self.window_size = window_size
        self.frame_idx = (data.index[0] + window_size, data.index[-1])
        self.shape = (window_size, data.shape[1]) # window_size * open, close, high, low, vol
        
        self.action_space = OneD(len(Actions)) #buy, hold, sell
        self.observation_space = Box(low = -np.inf, high = np.inf, size = self.shape, dtype = np.float64)
        
        self._start = self.window_size
        self._end = len(self.data.index)-1
        self.init_amount = init_amount
        self.prev = deque(maxlen=window_size)
        self.reset()
        
    # Resets the environment for each episode
    def reset(self):
        self.done = False
        self._current = self._start
        self.prev.extend([x for x in range(self.window_size)])
        self._position = Position.Short
        self._pos_hist = (self.window_size * [None])+[self._position]
        self._total_reward = 0
        self._total_profit = 0
        self.holdings = self.init_amount
        self.no_shares = 0
        self.hist = {}
        
    def _get_state(self):
        return self.data[(self._current - self.window_size +1): self._current+1]

    def step(self, action):
        self._current += 1
        
        if self._current == self._end:
            self.done = True
            
        self._update_profit(action)
        
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward
        
        trade = False
        if((action == Actions.Buy.value and self._position == Position.Short) or
           (action == Actions.Sell.value and self._position == Position.Long) or
           (action == Actions.Hold.value)):
            trade = True
            
        if trade:
            self._position = self._position.opposite()
            self.prev.append(self._current)
        
        self._pos_hist.append(self._position)
        step = self._get_state()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_hist(info)
        
        return step, step_reward, self.done, info
        
    def _update_hist(self, info):
        if not self.hist:
            self.hist = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.hist[key].append(value)
        
    def _update_profit(self, action):
        trade = False
        if((action == Actions.Buy.value and self._position == Position.Short) or
           (action == Actions.Sell.value and self._position == Position.Long)):
            trade = True
        
        if trade or self.done:
            current_price = self.data.loc[self._current, 'Close']
            prev_prices = []
            for idx in self.prev:
                prev_prices.append(self.data.loc[idx, 'Close'])
            
            if action == Actions.Sell.value:
                if self.no_shares > 0:
                    self.holdings = self.no_shares * current_price
                    # print(f'{self.holdings}, {current_price}, {self.no_shares}')
                    past_profit = []
                    for price in prev_prices:
                        past_profit.append(self.no_shares * price)
                    self._total_profit += ((self.no_shares * current_price) - max(past_profit))
                    self.no_shares = 0
            elif action == Actions.Buy.value:
                if self.holdings > 0 :
                    self.no_shares = self.holdings / current_price
                    # print(f'{self.holdings}, {current_price}, {self.no_shares}')
                    past_profit = []
                    for price in prev_prices:
                        past_profit.append(self.no_shares * price)
                    self._total_profit += (min(past_profit) - (self.no_shares * current_price))
                    self.holdings = 0
                
    def _calculate_reward(self, action):        
        if action == Actions.Buy.value or action == Actions.Sell.value:
            return int(self._total_profit - self.init_amount)
        else:
            return 0
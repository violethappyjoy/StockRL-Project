import numpy as np
from enum import Enum
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
    def __init__(self, data, window_size = 10):
        self.data = data
        self.window_size = window_size
        self.frame_idx = (data.index[0] + window_size, data.index[-1])
        self.shape = (window_size, data.shape[1]) # window_size * open, close, high, low, vol
        
        self.action_space = OneD(len(Actions)) #buy, hold, sell
        self.observation_space = Box(low = -np.inf, high = np.inf, size = self.shape, dtype = np.float64)
        
        self._start = self.window_size
        self._end = len(self.data.index)-1
        self.reset()
        
    # Resets the environment for each episode
    def reset(self):
        self.done = False
        self._current = self._start
        self._last = self._current-1
        self._position = Position.Short
        self._pos_hist = (self.window_size * [None])+[self._position]
        self._total_reward = 0
        self._total_profit = 1
        self.hist = {}
        
    def _get_state(self):
        return self.data[(self._current - self.window_size +1): self._current+1]
          
    # Three Actions: SELL{0}, HOLD{1}, BUY{3}
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
            self._last =  self._current
        
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
           (action == Actions.Sell.value and self._position == Position.Long) or
           (action == Actions.Hold.value)):
            trade = True
        
        if trade or self.done:
            current_price = self.data.loc[self._current, 'Close']
            prev_price = self.data.loc[self._last, 'Close']
            # print(f'{current_price}, {prev_price}, {action}, {self._position}')
            
            if self._position == Position.Long and action == Actions.Sell.value:
                shares = self._total_profit / prev_price
                self._total_profit = shares * current_price
            elif self._position == Position.Short and action == Actions.Buy.value:
                self._total_profit = self._total_profit * (current_price - prev_price) / prev_price
                
    def _calculate_reward(self, action):
        current_price = self.data.loc[self._current, 'Close']
        prev_price = self.data.loc[self._last, 'Close']
        shares = self._total_profit / prev_price
        
        if action == Actions.Buy.value:
            return shares * (current_price - prev_price)
        elif action == Actions.Sell.value:
            return shares * (prev_price - current_price)
        else:
            return 0
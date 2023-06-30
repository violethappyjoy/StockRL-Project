import numpy as np
from enum import Enum
from environment.env_var import Box, OneD

class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 3
    
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
        self.prices, self.signal = self._process_data()
        self.shape = (window_size, self.signal.shape[1]) # window_size * open, close, high, low, vol
        
        self.action_space = OneD(len(Actions)) #buy, hold, sell
        self.observation_space = Box(low = -np.inf, high = np.inf, size = self.shape, dtype = np.float64)
        
        self._start = self.window_size
        self._end = len(self.prices)-1
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
        
    def get_state(self):
        pass
        
    def sell_stock(self):
        pass
    
    def buy_stock(self):
        pass
    
    def hold_stock(self):
        pass
          
    # Three Actions: SELL{0}, HOLD{1}, BUY{3}
    def step(self, action):
        pass
    def _process_data(self):
        prices = self.data.loc[:,'Close'].to_numpy()
        assert len(prices) >= self.frame_idx[1], "ERROR: Insufficient data"
        # assert self.frame_idx[0] >= self.window_size or self.frame_idx[1], "ERROR: Index Out of Bound"
        
        prices = prices[self.frame_idx[0] - self.window_size: self.frame_idx[1]]
        
        diff = np.insert(np.diff(prices), 0, 0)
        mean = np.mean(prices)
        std = np.std(prices)
        
        multipier = 2.05
        if(self.window_size == 10):
            multipier = 1.9
        elif(self.window_size == 20):
            multipier = 2.0
        elif(self.window_size == 50):
            multipier = 2.1
        elif(self.window_size < 10):
            multipier = 1.85
        
        upper_band = mean + multipier * std
        lower_band = mean - multipier * std
        
        signal = np.column_stack((prices, diff, np.full_like(prices, upper_band),  np.full_like(prices, lower_band)))
        return prices, signal
                
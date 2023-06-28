import numpy as np
import pandas as pd

class MarketEnv:
    # initialize values for our Stock Market Environment
    # Data(DF), init_amt(initial money to buy), window size(amt of past data), timestamp
    def __init__(self, data, normalized_data, init_amt=1000, window_size=10, t=0):
        self.t = t
        self.init_amt = init_amt
        self.n_action = 3 #sell, hold, buy
        self.n_features = window_size * 5 # num of previous days * num of features types
        self.raw_data = data
        self.norm_data = normalized_data
        self.reset()
        
    # Resets the environment for each episode
    def reset(self):
        self.hold_amt = self.init_amt  # reset holding money to initial money
        self.buy_price = 0 # previous stock buying price
        self.sell_price = 0 # previous stock selling price
        self.nos_share = 0 # previous stock volume
        self.profit = 0 # value to calc reward
        self.reward = 0 # reward at current timestamp
        self.prev_acion = -1 # previous action
        
    def get_state(self):
        state = self.norm_data[self.t : self.t + self.window_size].values.flatten()
        return state 
        
    def sell_stock(self):
        if self.nos_share > 0:
            self.sell_price = self.raw_data.iloc[self.t]["Close"]
            self.hold_amt += self.sell_price * self.nos_share
            self.profit = (self.sell_price - self.buy_price) * self.nos_share
            self.nos_share = 0
    
    def buy_stock(self):
        if self.hold_amt > 0:
            self.buy_price = self.raw_data.iloc[self.t]["Close"]
            self.nos_share = self.hold_amt // self.buy_price
            self.hold_amt -= self.buy_price * self.nos_share
            self.profit = 0
    
    def hold_stock(self):
        pass
          
    # Three Actions: SELL{0}, HOLD{1}, BUY{3}
    def step(self, action):
        if action == 0:
            self.sell_stock()
        elif action == 1:
            self.hold_stock()
        elif action == 2:
            self.buy_stock()
            
        self.t += 1
        next_state = self.get_state()
        done = self.t >= len(self.norm_data)-1
        
        if self.profit > int(self.init_amt * 0.05):
            self.reward = 1
        else:
            self.reward = -1
        
        return next_state, self.reward, done
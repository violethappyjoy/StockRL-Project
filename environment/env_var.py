import numpy as np

class Box:
    def __init__(self, low, high, size, dtype = np.float64):
        self.low = low
        self.high = high
        self.shape = size
        self. dtype = dtype
        
    def sample(self):
        return np.random.uniform(low= self.low, high=self.high, size= self.shape).astype(self.dtype) # type: ignore
    def contains(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)
    def __repr__(self):
        return (f"low = {self.low}, high = {self.high}, shape = {self.shape}, dtype = {self.dtype}")
    
class OneD:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return np.random.randint(self.n)
    def contains(self, x):
        return 0 <= x <self.n
    def __repr__(self):
        return (f"Action_Space(n={self.n})")
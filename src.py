from utils import read_data, normalize_data, div_data
from environment.environment import MarketEnv

data=read_data("data/TSLA.csv")
test, train = div_data(data)
train = normalize_data(train)
test = normalize_data(test)

env = MarketEnv(train)

# print(train.head())
print(env.signal.shape[1])
from utils import read_data, normalize_data, div_data
from environment.environment import MarketEnv

data=read_data("data/TSLA.csv")
test, train = div_data(data)
# train = normalize_data(train)
test = normalize_data(test)

env = MarketEnv(train)
ctr = 0
while True:
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    print(f'{env.holdings}')
    print(f'info: {info}')
    ctr+=1
    if done:
        print(f'info: {info}')
        break
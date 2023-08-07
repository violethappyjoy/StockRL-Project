from tqdm import tqdm
import numpy as np
import random
import tensorflow as tf
import os
import time
from utils import read_data, normalize_data, div_data
from environment.envi import MarketEnv
from agent.agent import Agent, ModifiedTensorBoard

data=read_data("data/TSLA.csv")
test, train = div_data(data)
train = normalize_data(train)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')
    
ep_rewards = [-200]
# print(data.head())

env = MarketEnv(train)
env.reset()

EPISODES = 10
MODEL_NAME = 'STOCK_16C1DX16C1DX8D'

epsilon = 1
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.001

AGGREGATE_STATS_EVERY = 1
MIN_REWARD = -20


# while True:
#     action = env.action_space.sample()
#     # print(action)
#     n_state, reward, done, info = env.step(action)
#     # print(f'info: {info}')
#     if done:
#         print(f'info: {info}')
#         break

# actions = [0,1,2,1,1,1,1,2,0,1,0,0,0]
# env._current = 123
# env.render(actions)
    
agent = Agent(env)
actions = []
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    agent.tensorboard.step = episode
    actions.clear()
    
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)
            
        # actions.clear()
        actions.append(action)
            
        new_state, reward, done, info = env.step(action)
        
        episode_reward += reward
        # print(f"Current state before update_replay_memory: {current_state}")
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        
        current_state = new_state
        step += 1
        
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        show_colorbar = False
        if episode == 1:
            show_colorbar = True
        else:
            show_colorbar = False
        env.render(actions, show_colorbar)
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            
    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY
        EPSILON = max(EPSILON_MIN, epsilon)
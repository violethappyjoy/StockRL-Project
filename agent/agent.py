import keras
from keras.models import Sequential
import tensorflow.keras.backend as backend
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import numpy as np
import random
import time

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1_000
MIN_REPLAY_MEMORY_SIZE = 100
MODEL_NAME = 'STOCK_64X32X64D'
UPDATE_TARGET_EVERY = 5
MINIBATCH_SIZE = 100

EPISODES = 10_000

EPSILON = 1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class Agent:
    def __init__(self, env):
        self.env = env
        self.model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir = f'logs/{MODEL_NAME}-{int(time.time())}')
        
        self.target_update_counter = 0
        
    def create_model(self):
        model = Sequential()
        
        model.add(Conv2D(64, kernel_size=(3,3), input_shape=self.env.observation_space.shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(32, kernel_size=(3,3), input_shape=self.env.observation_space.shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
            
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
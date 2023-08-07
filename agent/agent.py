import tensorflow.keras as keras
from keras.models import Sequential
import tensorflow.keras.backend as backend
from tensorflow.python.ops.summary_ops_v2 import create_file_writer
from keras.layers import Dense, Conv1D, Flatten, Dropout, Activation, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import numpy as np
import random
import time

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = 'STOCK_16X16X8D_MAXPOOLING_FIXED'
UPDATE_TARGET_EVERY = 2
MINIBATCH_SIZE = 64


class ModifiedTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.step = 0
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.writer.flush()

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
        self.step += 1

class Agent:
    def __init__(self, env):
        self.env = env
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir = f'logs/{MODEL_NAME}-{int(time.time())}')
        
        self.target_update_counter = 0
        
    def create_model(self):
        model = Sequential()
    
        model.add(Conv1D(16, kernel_size=3, input_shape=self.env.observation_space.shape))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
    
        model.add(Conv1D(16, kernel_size=3))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
    
        model.add(Flatten())
        model.add(Dense(8))
        model.add(Dense(self.env.action_space.n, activation='linear'))
    
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        model.summary()
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
            
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # print(minibatch)
        
        current_states = np.array([transition[0] for transition in minibatch])
        # print(current_states)
        current_qs_list = self.model.predict(current_states)
        # print(current_qs_list)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        # print(future_qs_list)
        
        X = []
        Y = []
        
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_fututre_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_fututre_q
            else:
                new_q = reward
                
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            Y.append(current_qs)
            
        self.model.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, shuffle=False, verbose=0, callbacks=[self.tensorboard] if terminal_state else None)
            
        if terminal_state:
            self.target_update_counter += 1
                
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
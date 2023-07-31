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



def create_model(self):
    model = Sequential()
    
    # First Conv1D layer with strided convolution (stride=2)
    model.add(Conv1D(16, kernel_size=3, strides=2, input_shape=self.env.observation_space.shape))
    model.add(Activation('relu'))
    
    # Second Conv1D layer with strided convolution (stride=2)
    model.add(Conv1D(16, kernel_size=3, strides=2))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dense(self.env.action_space.n, activation='linear'))
    
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()
    return model

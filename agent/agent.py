import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import numpy as np
import time

REPLAY_MEMORY_SIZE = 50_000


class Agent:
    pass
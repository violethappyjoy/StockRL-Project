import pandas as pd
import numpy as np
from utils import read_data, normalize_data, div_data

data=read_data("data/TSLA.csv")
test, train = div_data(data)
datan=normalize_data(data)
print(data.head())
print(datan.head())
print(train.head())
print(test.head())
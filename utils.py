import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.set_index('Date')
    data = data.sort_index()
    return data

def normalize_data(data):
    #  the scaled values for each column using the min-max scaling formula
    return data.iloc[:,1:].apply(lambda x : (x - min(x)) / (max(x) - min(x)))

def div_data(data):
    times = sorted(data.index.values)
    test_idx = times[-int(0.05*len(times))] # index for 5% of ending data
    test = data[(data.index >= test_idx)]
    train =  data[(data.index < test_idx)]
    return test, train
        
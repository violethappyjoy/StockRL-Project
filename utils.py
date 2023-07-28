import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def read_data(filename):
    data = pd.read_csv(filename)
    assert data.ndim ==2, 'ERROR: Invalid Input'
    findata = data.drop('Date', axis=1)
    # data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    # data = data.set_index('Date')
    data = data.sort_index(ascending=False)
    findata = data.drop('Date', axis=1)
    return findata

def normalize_data(data):
    columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Scale the selected columns using min-max scaling
    scale = MinMaxScaler(feature_range=(0, 1))
    data.loc[:, columns_to_normalize] = scale.fit_transform(data.loc[:, columns_to_normalize])

    return data

    # return data.iloc[:,1:].apply(lambda x : (x - min(x)) / (max(x) - min(x)))

def div_data(data):
    times = sorted(data.index.values)
    test_idx = times[-int(0.05*len(times))] # index for 5% of ending data
    test = data[(data.index >= test_idx)]
    train =  data[(data.index < test_idx)]
    return test, train
        
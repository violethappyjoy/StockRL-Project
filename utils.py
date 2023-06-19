import pandas as pd

def read_data(filename):
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['Date'],format='%Y%m%d')
    data = data.set_index('Date')
    return data
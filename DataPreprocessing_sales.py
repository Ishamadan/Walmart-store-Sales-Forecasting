# change date parameter format
# fill all nan values
# sort values by store and date

import pandas as pd
def data_preprocess(data):
    data.Date = pd.to_datetime(data.Date, format='%d-%m-%Y')
    data.fillna(0, inplace=True)

   # data = data.sort_values(by=['Store', 'Date'])
    #data = data.reset_index(drop=True)
    return data

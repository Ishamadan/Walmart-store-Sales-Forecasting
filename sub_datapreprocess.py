# that will be for saving database
# change date parameter format
# fill all nan values
import pandas as pd

class DataPreprocess:

    def __init__(self):
        pass
    def datapreprocess_feature(self,features):
        features.fillna(0, inplace=True)

        features.Date = pd.to_datetime(features.Date, format='%Y-%m-%d')

        return features

    def datapreprocess_sale(self,sales):
        sales.Date = pd.to_datetime(sales.Date, format='%Y-%m-%d')
        sales.fillna(0, inplace=True)

        # data = data.sort_values(by=['Store', 'Date'])
        # data = data.reset_index(drop=True)
        return sales

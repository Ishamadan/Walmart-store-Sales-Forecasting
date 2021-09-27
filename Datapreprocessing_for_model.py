import pandas as pd
import pymysql
from sub_datapreprocess import DataPreprocess

class DataPreprocessor:
    def __init__(self,s):
        self.store = s
        conn = pymysql.connect(host='localhost',
                               port=3306,
                               user='root',
                               passwd='password',
                               db='retaildataset',
                               charset='utf8')

        sql ="select * from Stores_dataset where store='" +str(self.store) + "'"
        self.stores = pd.read_sql(sql,conn)

    def datapreprocessing(self,sales,features):
        dp = DataPreprocess()
        final = dp.datapreprocess_sale(sales)
        final = final.merge(dp.datapreprocess_feature(features), how="left", on=['Store', 'Date', 'IsHoliday'])
        final = final.merge(self.stores, how="left", on=['Store'])
        final.fillna(0,inplace=True)
        def getbool(a):
            if a == True:
                return 1
            else:
                return 0

        def super_bowl(a):
            if str(a)[:10] == '2010-02-12' or str(a)[:10] == '2011-02-11' or str(a)[:10] == '2012-02-10' or str(a)[
                                                                                                            :10] == '2013-02-08':
                return 1
            else:
                return 0

        def labor_day(a):
            if str(a)[:10] == '2010-09-10' or str(a)[:10] == '2011-09-11' or str(a)[:10] == '2012-09-07' or str(a)[
                                                                                                            :10] == '2013-09-06':
                return 1
            else:
                return 0

        def thanksgiving(a):
            if str(a)[:10] == '2010-11-26' or str(a)[:10] == '2011-11-25' or str(a)[:10] == '2012-11-23' or str(a)[
                                                                                                            :10] == '2013-11-29':
                return 1
            else:
                return 0

        def christmas(a):
            if str(a)[:10] == '2010-12-31' or str(a)[:10] == '2011-12-30' or str(a)[:10] == '2012-12-28' or str(a)[
                                                                                                            :10] == '2013-12-27':
                return 1
            else:
                return 0

        final['post_holiday'] = final.IsHoliday.shift(1)
        final['pred_holiday'] = final.IsHoliday.shift(-1)

        final["IsHoliday"] = final.IsHoliday.apply(getbool)
        final["post_holiday"] = final.post_holiday.apply(getbool)
        final["pred_holiday"] = final.pred_holiday.apply(getbool)

        final['super_bowl'] = final.Date.apply(super_bowl)
        final['labor_day'] = final.Date.apply(labor_day)
        final['thanksgiving'] = final.Date.apply(thanksgiving)
        final['christmas'] = final.Date.apply(christmas)

        final = pd.get_dummies(final, columns=["Type"])

        # Making Avg MarkDown
        final['AvgMarkDown'] = final['MarkDown1'] + final['MarkDown2'] + final['MarkDown3'] + final['MarkDown4'] + final[
        'MarkDown5']
        final['AvgMarkDown'] = final['AvgMarkDown'] / 5
        final['week'] = final.Date.dt.week
        final['year'] = final.Date.dt.year

        final['year_week'] = final['year'].astype(str) + final['week'].astype(str)
        final.loc[final['week'] < 10, 'year_week'] = final.loc[final['week'] < 10, 'year'].astype(str) + '0' + final.loc[final['week'] < 10, 'week'].astype(str)
        final['year_week'] = final['year_week'].astype(int)

        #final['store'] = final['Store']
        #final['dept'] = final['Dept']

        #final = pd.get_dummies(final, columns=["Store","Dept"])

        final = final.sort_values(by=['Store', 'Date'])
        final = final.reset_index(drop=True)

        others = final.filter(['Date','Dept'])

        #store = final.store
        #department = final.dept

        del final['Date']
        del final['MarkDown1']
        del final['MarkDown2']
        del final['MarkDown3']
        del final['MarkDown4']
        del final['MarkDown5']
        #del final['store']
        #del final['dept']
        print(final.isnull().sum(),others.isnull().sum())

        return others,final

import pymysql
con = pymysql.connect(host='localhost',
                                  port=3306,
                                  user='root',
                                  passwd='password',
                                  db='retaildataset',
                                  charset='utf8')
cur = con.cursor()
#query = "UPDATE Predict_last_update SET p_date='2011-12-30' where store=30"
query = "select * from Predict_dataset where Store=30"
#query = "delete from Predict_dataset where Store=30"
cur.execute(query)
#con.commit()
print(cur.fetchall())
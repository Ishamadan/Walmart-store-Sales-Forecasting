from flask import Flask, render_template, request,redirect,url_for,session,flash
import pymysql

app = Flask(__name__)

#Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# No cacheing at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/sign')
def signin_user():
    return render_template('sign.html')


@app.route('/login_developer')
def login_dev():
    return render_template('login_developer.html')


@app.route('/login_company')
def login_mang():
    return render_template('login_company.html')


@app.route('/login_store')
def login_user():
    return render_template('login_store.html')


@app.route('/insert_store', methods=['POST', 'GET'])
def insert_store():
    s=0
    if request.method == 'POST':
        try:
            name = request.form['name']
            store_id = request.form['store']
            email = request.form['email']
            email = email.lower()
            password = request.form['password']
            if (0 < int(store_id[2:]) < 46):
                # open connection to the database
                con = pymysql.connect(host='localhost',
                                       port=3306,
                                       user='root',
                                       passwd='password',
                                       db='sales_users',
                                       charset='utf8')
                cur = con.cursor()
                cur.execute("INSERT INTO Store (store_id,s_email,s_name,s_password) VALUES(%s, %s, %s, %s)",(store_id,email,name,password) )
                con.commit()
                msg = "You are registered successfully"
            else:
                msg = "Please enter correct Store id"
        except:
            con.rollback()
            msg = "Please contact to company"

        finally:
            return render_template("sign.html", msg=msg)
            cur.close()
            con.close()


@app.route('/verify_company',methods=['POST','GET'])
def verify_company():
    if request.method == 'POST':
        try:
            c=0
            company_id = request.form['company']
            ps = request.form['password']
            con = pymysql.connect(host='localhost',
                                   port=3306,
                                   user='root',
                                   passwd='password',
                                   db='sales_users',
                                   charset='utf8')
            cur = con.cursor()
            query = "select * from Company where comapany_id='" + company_id + "' and c_password='" + ps + "'"
            cur.execute(query)
            r = cur.fetchone()
            if r is None:
                msg = "Enter correct password or company id"
            else:
                msg = "logged in successfully...."
                c=1
        except:
            msg = 'Invalid id or password'
        finally:
            cur.close()
            con.close()
            if c==1:
                return redirect("/company_portal")
            else:
                return render_template("login_company.html",msg=msg)


@app.route('/verify_developer',methods=['POST','GET'])
def verify_developer():
    if request.method == 'POST':
        try:
            d=0
            developer_id = request.form['developer']
            ps = request.form['password']
            con = pymysql.connect(host='localhost',
                                  port=3306,
                                  user='root',
                                  passwd='password',
                                  db='sales_users',
                                  charset='utf8')
            cur = con.cursor()
            query = "select * from Developer where developer_id='" + developer_id + "' and d_password='" + ps + "'"
            cur.execute(query)
            r = cur.fetchone()
            if r is None:
                msg = "Enter correct password or developer id"
            else:
                msg = "logged in successfully...."
                d=1
        except:
            msg = 'Invalid id or password'
        finally:
            cur.close()
            con.close()
            return render_template("login_developer.html",msg=msg)


@app.route('/verify_store',methods=['POST','GET'])
def verify_store():
    if request.method == 'POST':
        try:
            s=0
            email = request.form['email']
            ps = request.form['password']
            con = pymysql.connect(host='localhost',
                                  port=3306,
                                  user='root',
                                  passwd='password',
                                  db='sales_users',
                                  charset='utf8')
            cur = con.cursor()
            query = "select * from Store where s_email='" + email + "' and s_password='" + ps + "'"
            cur.execute(query)
            r = cur.fetchone()
            if r is None:
                msg = "Enter correct password or email id"
            else:
                msg=r[2]
                s=1
        except:
            msg = 'Invalid email id or password'
        finally:
            cur.close()
            con.close()
            if s==1:
                session['Store_UName'] = r[2]
                session['Store_ID'] = r[0]
                flash("You are logged in successfully !")
                return redirect('/fetch_dept')


            else:
                return render_template("login_store.html",msg=msg)


@app.route('/fetch_dept')
def dept_fetch():
    con = pymysql.connect(host='localhost',
                          port=3306,
                          user='root',
                          passwd='password',
                          db='retaildataset',
                          charset='utf8')
    '''cur = con.cursor()
    cur.execute("select dept from Store_dept where store='" + Store_ID[2:] + "'")
    dpt=cur.fetchall()
    for d in dpt:
        departments.append(d[0])'''
    cur2 = con.cursor()
    cur2.execute("select u_date from Sales_last_update where store='" + session['Store_ID'][2:] + "'")
    u_date = cur2.fetchone()
    session['u_date'] = u_date[0]
    cursor1 = con.cursor()
    cursor1.execute("Select p_date from Predict_last_update where store='" + session['Store_ID'][2:] + "'")

    p_date = cursor1.fetchone()
    session['p_date'] = p_date[0]
    session['p_dates']=[]
    session['u_dates'] =[]
    cursor1.close()
    cur2.close()
    con.close()
    return redirect('/store_portal')


@app.route('/store_portal')
def s_home():
    from plotly.offline import plot
    from plotly.graph_objs import Scatter
    import plotly.graph_objs as go
    from flask import Markup
    try:
        conn = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           passwd='password',
                           db='retaildataset',
                           charset='utf8')
        cursor2 = conn.cursor()
        cursor2.execute("select Date,sum(Weekly_Sales) from Sales_dataset where Store='" + session['Store_ID'][2:] + "' group by Date")
        r = cursor2.fetchall()
        date = []
        sale = []
        for i in r:
            date.append(i[0])
            sale.append(i[1])
        fig = go.Figure({
            'data': [
                {'x': date, 'y': sale, 'mode': 'lines+markers', },
            ],
            'layout': {
                'title': 'Sales Data Visualization',
                'xaxis': {
                    'title': 'date',
                    'titlefont' :{
                             'family':'Courier New, monospace',
                             'size':18,
                     }
                },
                'yaxis': {
                    'title': 'weekly sale',
                    'titlefont': {
                        'family': 'Courier New, monospace',
                        'size': 18,
                    }
                },
            }
            })
        my_plot_div = plot(fig,output_type='div')
        return render_template('store_portal.html',
                               div_placeholder=Markup(my_plot_div),msg=session['Store_UName'])
    except:
        pass
    finally:
        cursor2.close()
        conn.close()


@app.route('/add_store_form')
def add_store_form():
    return render_template('add_store.html')


@app.route('/company_portal',methods=['POST','GET'])
def c_home():
    from plotly.offline import plot
    from plotly.graph_objs import Scatter
    import plotly.graph_objs as go
    from flask import Markup
    store = 'SA1'

    if request.method == "POST":
        #store=request.args.get['tvalue']
        store = (request.form['tvalue'])
    try:
        conn = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           passwd='password',
                           db='retaildataset',
                           charset='utf8')
        cursor2 = conn.cursor()
        cursor2.execute("select Date,sum(Weekly_Sales) from Sales_dataset where Store='" + store[2:] + "' group by Date")
        r = cursor2.fetchall()
        date = []
        sale = []
        for i in r:
            date.append(i[0])
            sale.append(i[1])
        fig = go.Figure({
            'data': [
                {'x': date, 'y': sale, 'mode': 'lines+markers', },
            ],
            'layout': {
                'title': 'Sales Data Visualization for store '+store+'',
                'xaxis': {
                    'title': 'date',
                    'titlefont': {
                        'family': 'Courier New, monospace',
                        'size': 18,
                    }
                },
                'yaxis': {
                    'title': 'weekly sale',
                    'titlefont': {
                        'family': 'Courier New, monospace',
                        'size': 18,
                    }
                },
            }
        })
        my_plot_div = plot(fig, output_type='div')
        cursor2.close()
        conn.close()
        return render_template('company_portal.html',
                           div_placeholder=Markup(my_plot_div))
    except:
        pass
    finally:
        pass


@app.route('/add_store',methods=['POST','GET'])
def add_store():
    if request.method == 'POST':
        try:
            name = request.form['name']
            store_id = request.form['store_id']
            con = pymysql.connect(host='localhost',
                                  port=3306,
                                  user='root',
                                  passwd='password',
                                  db='sales_users',
                                  charset='utf8')
            cur = con.cursor()
            cur.execute("INSERT INTO store_registered (store_name,store_id) VALUES(%s, %s)",(name,store_id) )
            con.commit()
            msg = "The Store is registered successfully"
        except:
            con.rollback()
            msg = "Already Registered/ Failed"

        finally:
            return render_template("add_store.html", msg=msg)
            cur.close()
            con.close()


@app.route('/next-month-sale',methods=['POST','GET'])
def next_sale():
    store = 'SA1'

    if request.method == "POST":
        # store=request.args.get['tvalue']
        store = (request.form['tvalue'])
    try:
        import datetime
        conn = pymysql.connect(host='localhost',
                               port=3306,
                               user='root',
                               passwd='password',
                               db='retaildataset',
                               charset='utf8')
        cursor1 = conn.cursor()
        cursor2 = conn.cursor()
        cursor2.execute("Select p_date from Predict_last_update where store='" + store[2:] + "'")

        pre_date = cursor2.fetchone()
        pre_date = pre_date[0]
        pre_dates = []

        date_1 = datetime.datetime.strptime(str(pre_date), "%Y-%m-%d")
        end_date = date_1 + datetime.timedelta(days=0)
        pre_dates.append(end_date.date().strftime("%Y-%m-%d"))
        date_1 = datetime.datetime.strptime(str(pre_date), "%Y-%m-%d")
        end_date = date_1 + datetime.timedelta(days=-7)
        pre_dates.append(end_date.date().strftime("%Y-%m-%d"))
        date_1 = datetime.datetime.strptime(str(pre_date), "%Y-%m-%d")
        end_date = date_1 + datetime.timedelta(days=-14)
        pre_dates.append(end_date.date().strftime("%Y-%m-%d"))

        cursor1.execute(
            "select Dept,Date,Weekly_Sales from Predict_dataset where Store='" + store[2:] + "' and Date in ('" + pre_dates[
                0] + "','" + pre_dates[1] + "','" + pre_dates[2] + "')")
        r1 = cursor1.fetchall()

        import csv
        c = csv.writer(open('C:\\Users\\Isha Madan\\PycharmProjects\\final_year\\static\\Next target.csv', 'w'),
                       lineterminator='\n')
        c.writerow(('Department', 'Date', 'Next target Sale for store'+store+''))
        for f in r1:
            c.writerow(f)

        return render_template('Next_month_sale.html',data=r1,id=store)
    except:
        pass
    finally:
        pass


@app.route('/display-past-s')
def past():
    import plotly.plotly as py
    from plotly.offline import plot
    import plotly.graph_objs as go
    from flask import Markup
    conn = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           passwd='password',
                           db='retaildataset',
                           charset='utf8')
    cursor1 = conn.cursor()
    cursor1.execute(
        "select Sales_dataset.Dept,Sales_dataset.Date,Features_dataset.MarkDown1,Features_dataset.MarkDown2,Features_dataset.MarkDown3,Features_dataset.MarkDown4,Features_dataset.MarkDown5,Sales_dataset.Weekly_Sales from Sales_dataset left join Features_dataset on Sales_dataset.Date=Features_dataset.Date and Sales_dataset.Store=Features_dataset.Store where Sales_dataset.Store='" + session['Store_ID'][
                                                                                                                                                                                                                                                                                                                                                                                               2:] + "'")
    rows = cursor1.fetchall()
    d,date,m1,m2,m3,m4,m5,ws = [],[],[],[],[],[],[],[]
    for r in rows:
        d.append(r[0])
        date.append(r[1])
        m1.append(r[2])
        m2.append(r[3])
        m3.append(r[4])
        m4.append(r[5])
        m5.append(r[6])
        ws.append(r[7])
    cursor2 = conn.cursor()
    cursor2.execute("select * from Stores_dataset where Store='" + session['Store_ID'][2:] + "'")
    details = cursor2.fetchone()
    title = session['Store_UName']
    trace = go.Table(
        header=dict(values=['<b>Department<b>', '<b>Date<b>','<b>Markdown1<b>','<b>Markdown2<b>','<b>Markdown3<b>','<b>Markdown4<b>','<b>Markdown5<b>','<b>Weekly sale<b>'],
                    line=dict(color='#7D7F80'),
                    fill=dict(color='#a1c3d1'),
                    height=45,
                    font={'size':14},
                    align=['left','center'] ),
        cells=dict(values=[d,date,m1,m2,m3,m4,m5,ws],
                   line=dict(color='#7D7F80'),
                   fill=dict(color='#EDFAFF'),
                   height=40,
                   font={'size': 14},
                   align=['left'] * 5))

    layout = dict(width=1000, height=2300,filtering=True)
    data = [trace]
    fig = dict(data=data, layout=layout)
    my_plot_div = plot(fig, output_type='div')
    return render_template('past_sale.html',
                           div_placeholder=Markup(my_plot_div),id=session['Store_ID'],title=title, details=details)


@app.route('/display-past-sale',methods=['GET','POST'])
def past_sale():
        if request.method=='POST':
            d = request.form['tvalue']

        conn = pymysql.connect(host='localhost',
                               port=3306,
                               user='root',
                               passwd='password',
                               db='retaildataset',
                               charset='utf8')
        cursor1 = conn.cursor()
        cursor1.execute(
            "select Sales_dataset.Dept,Sales_dataset.Date,Features_dataset.MarkDown1,Features_dataset.MarkDown2,Features_dataset.MarkDown3,Features_dataset.MarkDown4,Features_dataset.MarkDown5,Sales_dataset.Weekly_Sales,Sales_dataset.IsHoliday from Sales_dataset left join Features_dataset on Sales_dataset.Date=Features_dataset.Date and Sales_dataset.Store=Features_dataset.Store where Sales_dataset.Store='" + session['Store_ID'][
                                                                                                                                                                                                                                                                                                                                                                                                          2:] + "'")
        r1 = cursor1.fetchall()

        cursor2 = conn.cursor()
        cursor2.execute("select * from Stores_dataset where Store='" + session['Store_ID'][2:] + "'")
        details = cursor2.fetchone()
        title = session['Store_UName']
        return render_template('past_sale.html', title=title, details=details, data=r1, id=session['Store_ID'],
                               )


# fetch date.. make list
def update_dates():
    import datetime

    con = pymysql.connect(host='localhost',
                          port=3306,
                          user='root',
                          passwd='password',
                          db='retaildataset',
                          charset='utf8')

    cur2 = con.cursor()
    cur2.execute("select u_date from Sales_last_update where store='" + session['Store_ID'][2:] + "'")
    u_date = cur2.fetchone()
    u_date = u_date[0]
    cur2.close()
    u_dates=[]

    date_1 = datetime.datetime.strptime(str(u_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=7)
    u_dates.append(end_date.date().strftime("%Y-%m-%d"))
    date_1 = datetime.datetime.strptime(str(u_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=14)
    u_dates.append(end_date.date().strftime("%Y-%m-%d"))
    date_1 = datetime.datetime.strptime(str(u_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=21)
    u_dates.append(end_date.date().strftime("%Y-%m-%d"))

    return u_date,u_dates


def prediction_dates():
    import datetime

    con = pymysql.connect(host='localhost',
                          port=3306,
                          user='root',
                          passwd='password',
                          db='retaildataset',
                          charset='utf8')

    cursor1 = con.cursor()
    cursor1.execute("Select p_date from Predict_last_update where store='" + session['Store_ID'][2:] + "'")

    p_date = cursor1.fetchone()
    p_date = p_date[0]
    p_dates = []

    date_1 = datetime.datetime.strptime(str(p_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=7)
    p_dates.append(end_date.date().strftime("%Y-%m-%d"))
    date_1 = datetime.datetime.strptime(str(p_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=14)
    p_dates.append(end_date.date().strftime("%Y-%m-%d"))
    date_1 = datetime.datetime.strptime(str(p_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=21)
    p_dates.append(end_date.date().strftime("%Y-%m-%d"))

    return p_date,p_dates


@app.route('/upload_sale')
def upload_sale():
    u_date,u_dates=update_dates()


    return render_template('upload.html',title=session['Store_UName'],id=session['Store_ID'][2:],last_date=u_date,dates=u_dates)


@app.route('/upload-in-database',methods=['POST','GET'])
def upload_next():
    u_date,u_dates = update_dates()
    msg='Failed'
    Store_ID=session['Store_ID']
    import pandas as pd
    if request.method == 'POST':
        # user defined exception:
        class MyException(Exception):
            def __init__(self, msg):
                self.a = msg
        try:

            conn = pymysql.connect(host='localhost',
                                   port=3306,
                                   user='root',
                                   passwd='password',
                                   db='retaildataset',
                                   charset='utf8')
            df1 = pd.read_csv(request.files.get('fileupload1'),encoding='utf-8')
            df2 = pd.read_csv(request.files.get('fileupload2'),encoding='utf-8')
            #x=len(departments) *3

            if (df1.shape != (3,8) or df2.shape[1] != 5):
                raise Exception("Format is Wrong")

            if (df2.columns.values.tolist() != ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']
                or df1.columns.values.tolist() != ['Store', 'Date', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday']):
                raise Exception("Either column name or order is wrong..error")

            if (df1[['Store','Date','IsHoliday']].isnull().any().any() == True or df2.isnull().any().any() == True):
                raise Exception("Some values are missing..error")
            #if set(df2['Dept']) != set(departments) :
            #    raise Exception("Some departments are missing...error")

            if set(df1['Store']) != {int(Store_ID[2:])} or set(df2['Store']) != {int(Store_ID[2:])}:
                raise Exception("Store number is not correct")
            if (set(df1['IsHoliday']) != {True,False} and set(df1['IsHoliday']) != {True} and set(df1['IsHoliday']) != {False})\
                    or (set(df2['IsHoliday']) != {True,False} and set(df2['IsHoliday']) != {True} and set(df2['IsHoliday']) != {False}):
                raise Exception("IsHoliday Values should be either TRUE or FALSE")

            if set(df1['Date']) != set(u_dates) or set(df2['Date']) != set(u_dates):
                raise Exception("Dates are not correct")
            cursor1 = conn.cursor()
            cursor2 = conn.cursor()
            cursor3 = conn.cursor()
            import math
            for index,row in df1.iterrows():
                if(math.isnan(row['MarkDown1'])):
                    row['MarkDown1'] = 0
                if (math.isnan(row['MarkDown2'])):
                    row['MarkDown2'] = 0
                if (math.isnan(row['MarkDown3'])):
                    row['MarkDown3'] = 0
                if (math.isnan(row['MarkDown4'])):
                    row['MarkDown4'] = 0
                if (math.isnan(row['MarkDown5'])):
                    row['MarkDown5'] = 0
                cursor1.execute(
                    "INSERT INTO Features_dataset (Store,Date,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,IsHoliday) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                    (int(row['Store']), row['Date'], float(row['MarkDown1']), float(row['MarkDown2']), float(row['MarkDown3']),
                     float(row['MarkDown4']), float(row['MarkDown5']), str(row['IsHoliday'])))

            for index, row in df2.iterrows():
                cursor2.execute(
                    "INSERT INTO Sales_dataset (Store,Dept,Date,Weekly_Sales,IsHoliday) VALUES(%s,%s,%s,%s,%s)",
                    (int(row['Store']), row['Dept'], row['Date'], row['Weekly_Sales'], str(row['IsHoliday'])))

            u_date = max(u_dates)
            cursor3.execute("UPDATE Sales_last_update SET u_date='"+u_date+"' where store='"+Store_ID[2:]+"'")
            conn.commit()

            u_date,u_dates=update_dates()
            cursor1.close()
            cursor2.close()
            cursor3.close()
            msg = "File is uploaded successfully !"
            conn.close()
        except Exception as m:
            conn.rollback()
            conn.close()
            msg = m
        except:
            msg="Something is wrong"
        finally:


            return render_template("upload.html", msg=msg, title=session['Store_UName'],id=session['Store_ID'][2:], last_date=u_date,dates=u_dates)


@app.route('/predict_check')
def pre_check():
    msg = ''

    class MyException(Exception):
        def __init__(self, msg):
            self.a = msg
    try:
        p_date,p_dates = prediction_dates()
        u_date,u_dates = update_dates()

        if p_date > u_date :
            raise Exception("Upload sales data first")

        prediction_dates()
        return render_template('predict_fileupload.html', msg=msg, title=session['Store_UName'],id=session['Store_ID'][2:],last_predict=p_date,predict_for_dates=p_dates)
    except Exception as s:
        msg = s
        return render_template('404.html', msg=msg, title=session['Store_UName'],last_predict=p_date,last_upload=u_date)
    except:
        msg = "Something is wrong"


def predict(df1,df2):
    from training_store_wise import Train
    store = session['Store_ID'][2:]
    t = Train(store)
    print('in1')
    import pymysql
    import pandas as pd
    conn = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           passwd='password',
                           db='retaildataset',
                           charset='utf8')
    sql1 = "select * from Sales_dataset where Store='" + store + "'"
    sql2 = "select * from Features_dataset where Store='" + store + "'"
    # stores = psql.frame_query(sql,cursor2)
    print('in2')
    sales_data = pd.read_sql(sql1, conn)
    features_data = pd.read_sql(sql2,conn)
    y_pred=t.train_predict(sales_data,features_data,df2,df1)
    print('in3')
    return y_pred


@app.route('/upload-predict',methods=['GET','POST'])
def upload_pred():
    from datetime import datetime
    p_date,p_dates = prediction_dates()
    Store_ID=session['Store_ID']
    msg = 'Failed'
    import pandas as pd
    s =0
    if request.method == 'POST':
        # user defined exception:
        class MyException(Exception):
            def __init__(self, msg):
                self.a = msg

        try:
            conn = pymysql.connect(host='localhost',
                                   port=3306,
                                   user='root',
                                   passwd='password',
                                   db='retaildataset',
                                   charset='utf8')
            df1 = pd.read_csv(request.files.get('fileupload1'), encoding='utf-8')
            df2 = pd.read_csv(request.files.get('fileupload2'), encoding='utf-8')
            # x=len(departments) *3
            if (df1.shape != (3, 8) or df2.shape[1] != 4):
                print(df2.shape[1])
                raise Exception("Format is Wrong...")

            if (df2.columns.values.tolist() != ['Store', 'Dept', 'Date', 'IsHoliday']
                or df1.columns.values.tolist() != ['Store', 'Date', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                                                   'MarkDown5', 'IsHoliday']):
                raise Exception("Either column name or order is wrong..error")

            if (df1[['Store', 'Date', 'IsHoliday']].isnull().any().any() == True or df2.isnull().any().any() == True):
                raise Exception("Some values are missing..error")

            # if set(df2['Dept']) != set(departments) :
            #    raise Exception("Some departments are missing...error")

            if set(df1['Store']) != {int(Store_ID[2:])} or set(df2['Store']) != {int(Store_ID[2:])}:
                raise Exception("Store number is not correct")

            if (set(df1['IsHoliday']) != {True, False} and set(df1['IsHoliday']) != {True} and set(
                    df1['IsHoliday']) != {False}) \
                    or (set(df2['IsHoliday']) != {True, False} and set(df2['IsHoliday']) != {True} and set(
                        df2['IsHoliday']) != {False}):
                raise Exception("Values should be either TRUE or FALSE")

            if set(df1['Date']) != set(p_dates) or set(df2['Date']) != set(p_dates):
                raise Exception("Dates are not correct")
            print('Out')
            prediction = predict(df1,df2)
            print('out2')
            cursor1 = conn.cursor()
            cursor3 = conn.cursor()
            #print(prediction['Date'])
            #print(type(prediction['Date']))
            for index, row in prediction.iterrows():
                cursor1.execute(
                    "INSERT INTO Predict_dataset (Store,Dept,Date,Weekly_Sales) VALUES(%s,%s,%s,%s)",
                    (int(Store_ID[2:]), int(row['Dept']),str(row['Date']), float(row['Weekly_Sales'])))
            print('done1')
            #final = prediction.merge(df1, how="left", on='Date')

            p_date = max(p_dates)
            cursor3.execute("UPDATE Predict_last_update SET p_date='"+p_date+"' where store='"+Store_ID[2:]+"'")
            print('done2')
            conn.commit()
            p_date,p_dates = prediction_dates()

            cursor3.close()
            cursor1.close()

            #print(final)
            msg = "Success"
            conn.close()
            s=1
        except Exception as m:
            conn.rollback()
            conn.close()
            msg = m
            s=0
        except:
            msg = "Something is wrong"
            s=0
        finally:

            if s==0:
                return render_template("predict_fileupload.html", msg=msg, title=session['Store_UName'],id=session['Store_ID'][2:], last_predict=p_date,predict_for_dates=p_dates)
            else:
                return render_template("predict_success.html",msg=msg,title=session['Store_UName'],id=session['Store_ID'][2:], last_predict=p_date)


@app.route('/model-track',methods=['GET','POST'])
def model_track():
    store = 'SA1'
    dept = '1'
    if request.method == 'POST':
        store = request.form['tvalue1']
        dept = request.form['tvalue2']
    try:
        import all_models
        import pymysql
        import pandas as pd
        conn = pymysql.connect(host='localhost',
                               port=3306,
                               user='root',
                               passwd='password',
                               db='retaildataset',
                               charset='utf8')
        sql1 = "select * from Sales_dataset where Store='" + store[2:] + "'"
        sql2 = "select * from Features_dataset where Store='" + store[2:] + "'"
        # stores = psql.frame_query(sql,cursor2)
        sales_data = pd.read_sql(sql1, conn)
        features_data = pd.read_sql(sql2, conn)
        mt = all_models.AllTrain(store[2:],dept)
        chart, acc= mt.train_predict(sales_data, features_data)
        from flask import Markup
        return render_template('all_algo.html', div_placeholder=Markup(chart),
                               accuracy=acc)
    except:
        pass



@app.route('/see-prediction-store')
def see_ps():
    import datetime
    conn = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           passwd='password',
                           db='retaildataset',
                           charset='utf8')
    cursor1 = conn.cursor()
    cursor2 = conn.cursor()
    cursor2.execute("Select p_date from Predict_last_update where store='" +session['Store_ID'][2:] + "'")

    pre_date = cursor2.fetchone()
    pre_date = pre_date[0]
    pre_dates = []

    date_1 = datetime.datetime.strptime(str(pre_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=0)
    pre_dates.append(end_date.date().strftime("%Y-%m-%d"))
    date_1 = datetime.datetime.strptime(str(pre_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=-7)
    pre_dates.append(end_date.date().strftime("%Y-%m-%d"))
    date_1 = datetime.datetime.strptime(str(pre_date), "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=-14)
    pre_dates.append(end_date.date().strftime("%Y-%m-%d"))

    cursor1.execute(
        "select Dept,Date,Weekly_Sales from Predict_dataset where Store='" +session['Store_ID'][2:] + "' and Date in ('" + pre_dates[0] + "','" + pre_dates[1] + "','" + pre_dates[2] + "')")
    r1 = cursor1.fetchall()
    cursor2 = conn.cursor()
    cursor2.execute("select * from Stores_dataset where Store='" + session['Store_ID'][2:] + "'")
    details = cursor2.fetchone()
    title = session['Store_UName']
    if r1 == ():
        return render_template("view_error.html", msg="Please do prediction first...", title=title,details=details,id=session['Store_ID'])

    import csv
    c=csv.writer(open('C:\\Users\\Isha Madan\\PycharmProjects\\final_year\\static\\predict.csv','w'),lineterminator='\n')
    c.writerow(('Department','Date','Next Weekly Sale'))
    for f in r1:
        c.writerow(f)

    return render_template('see_prediction_store.html', title=title, details=details, data=r1, id=session['Store_ID'],
                           )


@app.route('/compare_chart',methods=['POST','GET'])
def compare_chart():
    from plotly.offline import plot
    from plotly.graph_objs import Scatter
    import plotly.graph_objs as go
    from flask import Markup
    store = 'SA1'

    if request.method == "POST":
        #store=request.args.get['tvalue']
        dept = (request.form['tvalue'])
    try:
        conn = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           passwd='password',
                           db='retaildataset',
                           charset='utf8')
        dept = '1'
        if request.method == "POST":
            # store=request.args.get['tvalue']
            dept = (request.form['tvalue'])

        cursor3 = conn.cursor()
        cursor3.execute("select Date,sum(Weekly_Sales) from Sales_dataset where Store='" + session['Store_ID'][2:] + "'and Date>'2011-12-30' group by Date")
        r3 = cursor3.fetchall()
        date3 = []
        sale3 = []
        for i in r3:
            date3.append(i[0])
            sale3.append(i[1])

        cursor4 = conn.cursor()
        cursor4.execute("select Date,sum(Weekly_Sales) from Predict_dataset where Store='" + session['Store_ID'][
                                                                                           2:] + "' group by Date")
        r4 = cursor4.fetchall()
        if r4 ==():
            return render_template("compare_error.html",msg="Please do prediction first...",title=session['Store_UName'])
        else:
            date4 = []
            sale4 = []
            for i in r4:
                date4.append(i[0])
                sale4.append(i[1])

            fig = go.Figure({
                'data': [
                    {'x': date3, 'y': sale3, 'mode': 'lines+markers','name':'Actual Weekly Sale' },
                    {'x': date4, 'y': sale4, 'mode': 'lines+markers','name':'Predicted Weekly Sale' },
                ],
                'layout': {
                    'title': 'Sales Data Prediction compare Visualization for store '+session['Store_ID'][2:]+'',
                    'xaxis': {
                        'title': 'date',
                        'titlefont': {
                            'family': 'Courier New, monospace',
                            'size': 18,
                        }
                    },
                    'yaxis': {
                        'title': 'weekly sale',
                        'titlefont': {
                            'family': 'Courier New, monospace',
                            'size': 18,
                        }
                    },
                }
            })
            my_plot_div = plot(fig, output_type='div')

            cursor1 = conn.cursor()
            cursor1.execute("select Date,Weekly_Sales from Sales_dataset where Store='" + session['Store_ID'][
                                                                                               2:] + "' and Dept='" + dept + "' and Date>'2011-12-30' ")
            r1 = cursor1.fetchall()
            date1 = []
            sale1 = []
            for i in r1:
                date1.append(i[0])
                sale1.append(i[1])

            cursor2 = conn.cursor()
            cursor2.execute("select Date,Weekly_Sales from Predict_dataset where Store='" + session['Store_ID'][
                                                                                                 2:] + "' and Dept='" + dept + "'")
            r2 = cursor2.fetchall()
            date2 = []
            sale2 = []
            for i in r2:
                date2.append(i[0])
                sale2.append(i[1])

            fig1 = go.Figure({
                'data': [
                    {'x': date1, 'y': sale1, 'mode': 'lines+markers','name': 'Actual Weekly Sale' },
                    {'x': date2, 'y': sale2, 'mode': 'lines+markers','name':'Predicted Weekly Sale' },
                ],
                'layout': {
                    'title': 'Sales Data Prediction compare Visualization for dept ' + dept + '',
                    'xaxis': {
                        'title': 'date',
                        'titlefont': {
                            'family': 'Courier New, monospace',
                            'size': 18,
                        }
                    },
                    'yaxis': {
                        'title': 'weekly sale',
                        'titlefont': {
                            'family': 'Courier New, monospace',
                            'size': 18,
                        }
                    },
                }
            })
            my_plot_div1 = plot(fig1, output_type='div')


            return render_template('charts.html',div_placeholder=Markup(my_plot_div),div_placeholder1=Markup(my_plot_div1),msg=session['Store_UName'])
    except:
        pass
    finally:
        cursor3.close()
        cursor4.close()
        conn.close()


@app.route('/logout_store')
def logout():
   # remove the username from the session if it is there
   session.pop('Store_UName', None)
   session.pop('Store_ID', None)
   return redirect('/login_store')


#import base64, M2Crypto
#def generate_session_id(num_bytes = 16):
#    return base64.b64encode(M2Crypto.m2.rand_bytes(num_bytes))


if __name__ == '__main__':
    #app.secret_key = generate_session_id()
    app.run(debug=True)
    #app.run()



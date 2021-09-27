import numpy as np
import pandas as pd
from Datapreprocessing_for_model import DataPreprocessor


class AllTrain:
    def __init__(self, i,j):
        self.store = int(i)
        self.department = j
        self.dp = DataPreprocessor(self.store)
        pass

    def train_predict(self, sales_data, features_data):
        split_length = int(features_data.shape[0]*7/8)
        split_date_dev = features_data.loc[split_length,'Date']

        s_train = sales_data.loc[sales_data['Date'] <= split_date_dev]
        s_test = sales_data.loc[sales_data['Date'] > split_date_dev]
        f_train = features_data.loc[sales_data['Date'] <= split_date_dev]
        f_test = features_data.loc[sales_data['Date'] > split_date_dev]
        #s_train = sales_train.copy()
        #s_test = sales_test.copy()

        s_train = s_train[s_train.Store == self.store]
        s_test = s_test[s_test.Store == self.store]

        #f_train = features_train.copy()
        #f_test = features_test.copy()

        f_train = f_train[f_train.Store == self.store]
        f_test = f_test[f_test.Store == self.store]

        other_training, training_data = self.dp.datapreprocessing(s_train, f_train)
        other_testing, testing_data = self.dp.datapreprocessing(s_test, f_test)
        # print("Training processed data")
        # print(training_data)
        # print("--------------------------------------")
        # print(testing_data)
        #X_train = np.array(training_data.drop(['Weekly_Sales'], axis=1))
        #Y_train = np.array(training_data['Weekly_Sales'])

        #X_test = np.array(testing_data)
        other_training['Weekly_Sales'] = training_data['Weekly_Sales']
        X_test = np.array(testing_data.drop(['Weekly_Sales'], axis=1))
        Y_test = np.array(testing_data['Weekly_Sales'])

        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # scaler.transform(X_test)

        # from sklearn.ensemble import RandomForestRegressor
        # regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        # regressor.fit(X_train,Y_train)

        # from sklearn.linear_model import LinearRegression
        # regressor = LinearRegression()
        # regressor.fit(X_train,Y_train)

        # Fitting SVR to the dataset
        # from sklearn.svm import SVR
        # regressor = SVR(kernel = 'rbf')
        # regressor.fit(X_train,Y_train)

        # from sklearn.ensemble import ExtraTreesRegressor
        # regressor = ExtraTreesRegressor()
        # regressor.fit(X_train,Y_train)


        from sklearn.feature_selection import RFE
        from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
        from sklearn.model_selection import KFold

        #
        # regressor = KNeighborsRegressor(n_neighbors=10)
        # regressor.fit(X_train,Y_train)

        # from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        # regressor = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1, n_jobs=1)
        # regressor.fit(X_train, Y_train)

        def knn():
            from sklearn.neighbors import KNeighborsRegressor
            knn = KNeighborsRegressor(n_neighbors=10)
            return knn

        def lineaRegressor():
            from sklearn.linear_model import LinearRegression
            clf = LinearRegression()
            return clf

        def decisionTreeRegressor():
            from sklearn.tree import DecisionTreeRegressor
            clf = DecisionTreeRegressor(random_state=0)
            return clf

        def extraTreesRegressor():
            from sklearn.ensemble import ExtraTreesRegressor
            clf = ExtraTreesRegressor(n_estimators=100, max_features='auto', verbose=1, n_jobs=1)
            return clf

        def randomForestRegressor():
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(n_estimators=100, max_features='log2', verbose=1)
            return clf

        def svm():
            from sklearn.svm import SVR
            clf = SVR(kernel='rbf', gamma='auto')
            return clf

        def nn():
            from sklearn.neural_network import MLPRegressor
            clf = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', verbose=3)
            return clf

        def predict_(m, test_x):
            return pd.Series(m.predict(test_x))
        print("Algorithms   MAE   MSE   R2 Score")
        def model_():

            #     return knn()
            #return extraTreesRegressor()

            return lineaRegressor()
        #    return decisionTreeRegressor()
        #     return svm()
        #     return nn()
        #     return randomForestRegressor()

        def train_(train_x, train_y):
            m = model_()
            m.fit(train_x, train_y)
            return m

        def train_and_predict(train_x, train_y, test_x):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(train_x)
            scaler.transform(test_x)

            m = train_(train_x, train_y)
            return predict_(m, test_x), m

        def model1_():
            return knn()
        #    return lineaRegressor()


        #    return extraTreesRegressor()


        #    return decisionTreeRegressor()
        #     return svm()
        #     return nn()
        #     return randomForestRegressor()

        def train1_(train_x, train_y):
            m = model1_()
            m.fit(train_x, train_y)
            return m

        def train_and_predict1(train_x, train_y, test_x):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(train_x)
            scaler.transform(test_x)

            m = train1_(train_x, train_y)
            return predict_(m, test_x), m

        def model2_():
            return decisionTreeRegressor()

            #return knn()
            #return extraTreesRegressor()

        #    return lineaRegressor()

        #     return svm()
        #     return nn()
        #     return randomForestRegressor()

        def train2_(train_x, train_y):
            m = model2_()
            m.fit(train_x, train_y)
            return m

        def train_and_predict2(train_x, train_y, test_x):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(train_x)
            scaler.transform(test_x)

            m = train2_(train_x, train_y)
            return predict_(m, test_x), m

        def model3_():
            return randomForestRegressor()
            #return decisionTreeRegressor()

            #     return knn()
            #return extraTreesRegressor()

         #   return lineaRegressor()

        #     return svm()
        #     return nn()


        def train3_(train_x, train_y):
            m = model3_()
            m.fit(train_x, train_y)
            return m

        def train_and_predict3(train_x, train_y, test_x):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(train_x)
            scaler.transform(test_x)

            m = train3_(train_x, train_y)
            return predict_(m, test_x), m

        def model4_():
            return extraTreesRegressor()

            #     return knn()

        #    return lineaRegressor()
         #   return decisionTreeRegressor()
        #     return svm()
        #     return nn()
        #    return randomForestRegressor()

        def train4_(train_x, train_y):
            m = model4_()
            m.fit(train_x, train_y)
            return m

        def train_and_predict4(train_x, train_y, test_x):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(train_x)
            scaler.transform(test_x)

            m = train4_(train_x, train_y)
            return predict_(m, test_x), m

        def calculate_error(test_y, predicted, weights):
            return mean_absolute_error(test_y, predicted, sample_weight=weights)

        kf = KFold(n_splits=5)
        splited = []
        # dataset2 = dataset.copy()
        for name, group in training_data.groupby(["Store", "Dept"]):
            group = group.reset_index(drop=True)
            trains_x = []
            trains_y = []
            tests_x = []
            tests_y = []
            # print(group.shape[0])
            if group.shape[0] <= 5:
                f = np.array(range(5))
                np.random.shuffle(f)
                group['fold'] = f[:group.shape[0]]
                continue
            fold = 0
            for train_index, test_index in kf.split(group):
                group.loc[test_index, 'fold'] = fold
                fold += 1
            splited.append(group)

        splited = pd.concat(splited).reset_index(drop=True)

        best_model1 = None
        error_cv1 = 0
        best_error1 = np.iinfo(np.int32).max
        for fold in range(5):
            dataset_train = splited.loc[splited['fold'] != fold]
            dataset_test = splited.loc[splited['fold'] == fold]
            train_y = dataset_train['Weekly_Sales']
            train_x = dataset_train.drop(['Weekly_Sales', 'fold'], axis=1)
            test_y = dataset_test['Weekly_Sales']
            test_x = dataset_test.drop(['Weekly_Sales', 'fold'], axis=1)
            #print(dataset_train.shape, dataset_test.shape)
            predicted, model1 = train_and_predict(train_x, train_y, test_x)
            weights = test_x['IsHoliday'].replace(True, 5).replace(False, 1)
            error1 = calculate_error(test_y, predicted, weights)
            error_cv1 += error1
            #print(fold, error)
            if error1 < best_error1:
                #print('Find best model')
                best_error1 = error1
                best_model1 = model1
        error_cv1 /= 5


        y_pred1 = best_model1.predict(X_test)
        other_testing['Weekly_Sales1'] = y_pred1
        ac1 = best_model1.score(X_test, Y_test)
        #print(ac)
        print('Linear Regression   ',mean_absolute_error(Y_test,y_pred1),mean_absolute_error(Y_test,y_pred1),r2_score(Y_test,y_pred1))

        best_model2 = None
        error_cv2 = 0
        best_error2 = np.iinfo(np.int32).max
        for fold in range(5):
            dataset_train = splited.loc[splited['fold'] != fold]
            dataset_test = splited.loc[splited['fold'] == fold]
            train_y = dataset_train['Weekly_Sales']
            train_x = dataset_train.drop(['Weekly_Sales', 'fold'], axis=1)
            test_y = dataset_test['Weekly_Sales']
            test_x = dataset_test.drop(['Weekly_Sales', 'fold'], axis=1)
            # print(dataset_train.shape, dataset_test.shape)
            predicted, model2 = train_and_predict1(train_x, train_y, test_x)
            weights = test_x['IsHoliday'].replace(True, 5).replace(False, 1)
            error2 = calculate_error(test_y, predicted, weights)
            error_cv2 += error2
            # print(fold, error)
            if error2 < best_error2:
                # print('Find best model')
                best_error2 = error2
                best_model2 = model2
        error_cv2 /= 5

        y_pred2 = best_model2.predict(X_test)
        other_testing['Weekly_Sales2'] = y_pred2
        ac2 = best_model2.score(X_test, Y_test)
        print('KNN   ',mean_absolute_error(Y_test,y_pred2),mean_absolute_error(Y_test,y_pred2),r2_score(Y_test,y_pred2))


        best_model3 = None
        error_cv3 = 0
        best_error3 = np.iinfo(np.int32).max
        for fold in range(5):
            dataset_train = splited.loc[splited['fold'] != fold]
            dataset_test = splited.loc[splited['fold'] == fold]
            train_y = dataset_train['Weekly_Sales']
            train_x = dataset_train.drop(['Weekly_Sales', 'fold'], axis=1)
            test_y = dataset_test['Weekly_Sales']
            test_x = dataset_test.drop(['Weekly_Sales', 'fold'], axis=1)
            # print(dataset_train.shape, dataset_test.shape)
            predicted, model3 = train_and_predict2(train_x, train_y, test_x)
            weights = test_x['IsHoliday'].replace(True, 5).replace(False, 1)
            error3 = calculate_error(test_y, predicted, weights)
            error_cv3 += error3
            # print(fold, error)
            if error3 < best_error3:
                # print('Find best model')
                best_error3 = error3
                best_model3 = model3
        error_cv3 /= 5

        y_pred3 = best_model3.predict(X_test)
        other_testing['Weekly_Sales3'] = y_pred3
        ac3 = best_model3.score(X_test, Y_test)
        print('Decision Tree   ',mean_absolute_error(Y_test,y_pred3),mean_absolute_error(Y_test,y_pred3),r2_score(Y_test,y_pred3))



        best_model4 = None
        error_cv4 = 0
        best_error4 = np.iinfo(np.int32).max
        for fold in range(5):
            dataset_train = splited.loc[splited['fold'] != fold]
            dataset_test = splited.loc[splited['fold'] == fold]
            train_y = dataset_train['Weekly_Sales']
            train_x = dataset_train.drop(['Weekly_Sales', 'fold'], axis=1)
            test_y = dataset_test['Weekly_Sales']
            test_x = dataset_test.drop(['Weekly_Sales', 'fold'], axis=1)
            # print(dataset_train.shape, dataset_test.shape)
            predicted, model4 = train_and_predict3(train_x, train_y, test_x)
            weights = test_x['IsHoliday'].replace(True, 5).replace(False, 1)
            error4 = calculate_error(test_y, predicted, weights)
            error_cv4 += error4
            # print(fold, error)
            if error4 < best_error4:
                # print('Find best model')
                best_error4 = error4
                best_model4 = model4
        error_cv4 /= 5

        y_pred4 = best_model4.predict(X_test)
        other_testing['Weekly_Sales4'] = y_pred4
        ac4 = best_model4.score(X_test, Y_test)
        print('Random Forest Tree  ',mean_absolute_error(Y_test,y_pred4),mean_absolute_error(Y_test,y_pred4),r2_score(Y_test,y_pred4))


        best_model5 = None
        error_cv5 = 0
        best_error5 = np.iinfo(np.int32).max
        for fold in range(5):
            dataset_train = splited.loc[splited['fold'] != fold]
            dataset_test = splited.loc[splited['fold'] == fold]
            train_y = dataset_train['Weekly_Sales']
            train_x = dataset_train.drop(['Weekly_Sales', 'fold'], axis=1)
            test_y = dataset_test['Weekly_Sales']
            test_x = dataset_test.drop(['Weekly_Sales', 'fold'], axis=1)
            # print(dataset_train.shape, dataset_test.shape)
            predicted, model5 = train_and_predict4(train_x, train_y, test_x)
            weights = test_x['IsHoliday'].replace(True, 5).replace(False, 1)
            error5 = calculate_error(test_y, predicted, weights)
            error_cv5 += error5
            # print(fold, error)
            if error5 < best_error5:
                # print('Find best model')
                best_error5 = error5
                best_model5 = model5
        error_cv5 /= 5

        y_pred5 = best_model5.predict(X_test)
        other_testing['Weekly_Sales5'] = y_pred5
        ac5 = best_model5.score(X_test, Y_test)
        print('Extra Forest Tree  ',mean_absolute_error(Y_test,y_pred5),mean_absolute_error(Y_test,y_pred5),r2_score(Y_test,y_pred5))


        date2 = []
        for i in other_testing['Date']:
            date2.append(i)
        from plotly.offline import plot
        from plotly.graph_objs import Scatter
        from flask import Markup
        import plotly.graph_objs as go

        #actual = pd.DataFrame({'date':date2,'sale':Y_test})
        #actual = actual.groupby(['date']).sum()
        other_testing['Weekly_Sales'] = Y_test
        pred = other_testing[other_testing.Dept == int(self.department)]
        #pred1 = pd.DataFrame({'date':date2,'sale':y_pred1})
        #pred1 = pred1.groupby(['date']).sum()

        #pred2 = pd.DataFrame({'date': date2, 'sale': y_pred2})
        #pred2 = pred2.groupby(['date']).sum()
        #pred3 = pd.DataFrame({'date': date2, 'sale': y_pred3})
        #pred3 = pred3.groupby(['date']).sum()
        #pred4 = pd.DataFrame({'date': date2, 'sale': y_pred4})
        #pred4 = pred4.groupby(['date']).sum()
        #pred5 = pd.DataFrame({'date': date2, 'sale': y_pred5})
        #pred5 = pred5.groupby(['date']).sum()
        #print(other_training['Date'].shape,training_data['Weekly_Sales'].shape,y_pred.shape)
        fig = go.Figure({
            'data': [
                {'x': pred['Date'], 'y': pred['Weekly_Sales'], 'mode': 'lines+markers', 'name': 'Actual Weekly Sale'},
                {'x': pred['Date'], 'y': pred['Weekly_Sales1'], 'mode': 'lines+markers', 'name': 'Predicted Weekly Sale by Linear regression'},
                {'x': pred['Date'], 'y': pred['Weekly_Sales2'], 'mode': 'lines+markers', 'name': 'Predicted Weekly Sale by KNN'},
                {'x': pred['Date'], 'y': pred['Weekly_Sales3'], 'mode': 'lines+markers', 'name': 'Predicted Weekly Sale by Decision Tree'},
                {'x': pred['Date'], 'y': pred['Weekly_Sales4'], 'mode': 'lines+markers', 'name': 'Predicted Weekly Sale by Random forest Tree'},
                {'x': pred['Date'], 'y': pred['Weekly_Sales5'], 'mode': 'lines+markers','name': 'Predicted Weekly Sale by Extra Forest'},

            ],
            'layout': {
                'title': 'Sales Data Prediction compare Visualization for store ' + str(self.store) + ' ,dept '+str(self.department)+'',
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

        return my_plot_div, [ac1,ac2,ac3,ac4,ac5]



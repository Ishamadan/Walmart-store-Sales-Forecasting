import numpy as np
import pandas as pd
from Datapreprocessing_for_model import DataPreprocessor


class Train:
    def __init__(self, i):
        self.store = int(i)
        self.dp = DataPreprocessor(self.store)
        pass

    def train_predict(self, sales_train, features_train, sales_test, features_test):

        s_train = sales_train.copy()
        s_test = sales_test.copy()

        s_train = s_train[s_train.Store == self.store]
        s_test = s_test[s_test.Store == self.store]

        f_train = features_train.copy()
        f_test = features_test.copy()

        f_train = f_train[f_train.Store == self.store]
        f_test = f_test[f_test.Store == self.store]

        other_training, training_data = self.dp.datapreprocessing(s_train, f_train)
        other_testing, testing_data = self.dp.datapreprocessing(s_test, f_test)
        # print("Training processed data")
        # print(training_data)
        # print("--------------------------------------")
        # print(testing_data)
        X_train = np.array(training_data.drop(['Weekly_Sales'], axis=1))
        Y_train = np.array(training_data['Weekly_Sales'])

        #X_test = np.array(testing_data)
        X_test = np.array(training_data.drop(['Weekly_Sales'], axis=1))
        Y_test = np.array(training_data['Weekly_Sales'])

        # X_test = np.array(testing_data.drop(['Weekly_Sales'], axis=1))
        # Y_test = np.array(testing_data['Weekly_Sales'])

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
        from sklearn.metrics import mean_absolute_error
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

        def model_():

            #     return knn()
            return extraTreesRegressor()

        #    return lineaRegressor()
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

        #
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

        best_model = None
        error_cv = 0
        best_error = np.iinfo(np.int32).max
        for fold in range(5):
            dataset_train = splited.loc[splited['fold'] != fold]
            dataset_test = splited.loc[splited['fold'] == fold]
            train_y = dataset_train['Weekly_Sales']
            train_x = dataset_train.drop(['Weekly_Sales', 'fold'], axis=1)
            test_y = dataset_test['Weekly_Sales']
            test_x = dataset_test.drop(['Weekly_Sales', 'fold'], axis=1)
            print(dataset_train.shape, dataset_test.shape)
            predicted, model = train_and_predict(train_x, train_y, test_x)
            weights = test_x['IsHoliday'].replace(True, 5).replace(False, 1)
            error = calculate_error(test_y, predicted, weights)
            error_cv += error
            print(fold, error)
            if error < best_error:
                print('Find best model')
                best_error = error
                best_model = model
        error_cv /= 5

        # Predicting a new result
        y_pred = best_model.predict(X_test)
        ac = best_model.score(X_test,Y_test)
        # print(ac)

        y_pred = best_model.predict(X_test)
        other_testing['Weekly_Sales'] = y_pred
        return other_testing


# len(x_train)





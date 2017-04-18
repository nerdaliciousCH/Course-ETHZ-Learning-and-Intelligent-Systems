import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, PolynomialFeatures
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")

    # Load the data from the CSV files
    train_data = pd.read_csv('train.csv', header=0)
    test_data = pd.read_csv('test.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = np.asarray(train_data['y'])
    ID = train_data['Id']
    X = train_data.drop(['Id', 'y'], axis=1)
    t_id = test_data['Id']
    x_test = test_data.drop('Id', axis=1)

    # transform data
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    x_test = scaler.transform(x_test)

    '''
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    X = poly.fit_transform(np.array(X))
    x_test = poly.fit_transform(np.array(x_test))
    '''

    # split for validation testing
    x_train, x_validate, y_train, y_validate = train_test_split(X, Y)

    #X = normalize(X, axis=0)
    #x_test = normalize(x_test, axis=0)

    # This is your model that will learn to predict
    #model = linear_model.LinearRegression(n_jobs=-1)

    #alphas = np.logspace(-15, 100, 100)
    #model = linear_model.RidgeCV(alphas, cv=10)

    #model = linear_model.LassoCV(eps=0.001, n_alphas=1000, alphas=np.logspace(-100, 100, 1000), fit_intercept=True, normalize=False, precompute='auto', max_iter=10000, tol=0.0001, copy_X=True, cv=10, verbose=False, n_jobs=-1, positive=False, random_state=None, selection='cyclic')

    #SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)

    param_grid = [{'C':np.logspace(-3, 2, 500), 'kernel': ['rbf', 'linear']}]
    model = GridSearchCV(svm.SVC(max_iter=1e6), param_grid, cv=5, scoring=None, fit_params=None, n_jobs=6, iid=True, refit=True, verbose=1, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

    #model = svm.SVR(kernel='poly', degree=5, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

    print("Training...")
    model.fit(x_train, y_train)

    best_estimator = model.best_estimator_

    print 'Best score of Grid Search: ' + str(model.best_score_)
    print 'Best params of Grid Search: ' + str(model.best_params_)

    # print mean squared error as first estimate
    acc = accuracy_score(y_validate, model.predict(x_validate))
    print "accuracy_score:"
    print acc

    # Train on entire Set
    best_estimator.fit(X, Y)


    print("Predicting...")
    y_prediction = best_estimator.predict(x_test)


    results = y_prediction
    results_df = pd.DataFrame(data={'y':results})
    joined = pd.DataFrame(t_id).join(results_df)

    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
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


    # split for validation testing
    x_train, x_validate, y_train, y_validate = train_test_split(X, Y)

    #X = normalize(X, axis=0)
    #x_test = normalize(x_test, axis=0)

    # This is your model that will learn to predict
    #model = linear_model.LinearRegression(n_jobs=-1)

    #alphas = np.logspace(-15, 100, 100)
    #model = linear_model.RidgeCV(alphas, cv=10)

    #model = linear_model.LassoCV(eps=0.001, n_alphas=1000, alphas=np.logspace(-100, 100, 1000), fit_intercept=True, normalize=False, precompute='auto', max_iter=10000, tol=0.0001, copy_X=True, cv=10, verbose=False, n_jobs=-1, positive=False, random_state=None, selection='cyclic')

    # param_grid = [{'C':np.linspace(0.01, 100, 100),  'epsilon':np.linspace(0.01, 100, 100), 'kernel': ['poly'], 'degree':np.linspace(1, 5, 5)}]
    # model = GridSearchCV(svm.SVR(), param_grid, cv=10, scoring=None, fit_params=None, n_jobs=-1, iid=True, refit=True, verbose=2, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

    #model = svm.SVR(kernel='poly', degree=5, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

    print("Training...")
    # x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    # x_validate = (x_validate - np.mean(x_validate)) / np.std(x_validate)

    
    # demjan magic
    numFeatures = 15
    fullMask = 2 ** numFeatures

    bestScore = 100

    for mask in range(1, fullMask):
        if (mask % 1 == 0):
            print(mask, fullMask)
            print(bestScore)

        feature_indices = []
        for i in range(numFeatures):
            if (mask & (2 ** i)):
                feature_indices.append(i)

        x_train_selected = x_train[feature_indices]

        # model = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
        model = svm.SVR(kernel='poly', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=36.3699, epsilon=4.0499, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        model.fit(x_train_selected, y_train)

        # best_estimator = model.best_estimator_

        # print 'Best score of Grid Search: ' + str(model.best_score_)
        # print 'Best params of Grid Search: ' + str(model.best_params_)

        # print mean squared error as first estimate
        x_validate_selected = x_validate[feature_indices]
        RMSE = mean_squared_error(y_validate, model.predict(x_validate_selected))**0.5
        # print "Root median square error:"
        # print RMSE

        bestScore = min(bestScore, RMSE)

    print(bestScore)

    # Train on entire Set
    # best_estimator.fit(X, Y)


    print("Predicting...")
    # y_prediction = best_estimator.predict(x_test)


    # results = y_prediction
    # results_df = pd.DataFrame(data={'y':results})
    # joined = pd.DataFrame(t_id).join(results_df)

    # print("Writing predictions to predictions.csv")
    # # Save the predictions out to a CSV file
    # joined.to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()

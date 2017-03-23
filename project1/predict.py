import pandas as pd
import numpy as np
from sklearn import linear_model, grid_search, svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize


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

    # Exclude a internal train set
    idx = range(len(ID))
    np.random.shuffle(idx)

    border = int(len(idx) * 0.9)
    validation_idx = idx[border:]
    idx = idx[:border]

    X = np.array(X)

    validation_x = X[validation_idx]
    validation_id = ID[validation_idx]
    validation_y = Y[validation_idx]

    train_X = X[[idx]]
    train_Y = Y[idx]
    train_id = ID[idx]

    #X = normalize(X, axis=0)
    #x_test = normalize(x_test, axis=0)

    # This is your model that will learn to predict
    #model = linear_model.LinearRegression(n_jobs=-1)

    #alphas = np.logspace(-15, 100, 100)
    #model = linear_model.RidgeCV(alphas, cv=10)

    #model = linear_model.LassoCV(eps=0.001, n_alphas=1000, alphas=np.logspace(-100, 100, 1000), fit_intercept=True, normalize=False, precompute='auto', max_iter=10000, tol=0.0001, copy_X=True, cv=10, verbose=False, n_jobs=-1, positive=False, random_state=None, selection='cyclic')

    param_grid = [{'C':np.linspace(0.01, 100, 2),  'epsilon':np.linspace(0.01, 100, 2), 'kernel': ['poly'], 'degree':np.linspace(1, 5, 3)}]
    model = grid_search.GridSearchCV(svm.SVR(), param_grid, cv=10)

    #model = svm.SVR(kernel='poly', degree=5, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

    print("Training...")
    model.fit(train_X, train_Y)

    best_estimator = model.best_estimator_

    print 'Best score of Grid Search: ' + str(model.best_score_)
    print 'Best params of Grid Search: ' + str(model.best_params_)

    RMSE = mean_squared_error(validation_y, model.predict(validation_x))**0.5
    print RMSE

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

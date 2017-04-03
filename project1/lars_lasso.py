import pandas as pd
import numpy as np
from sklearn import linear_model, grid_search
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize, PolynomialFeatures
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
    poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)
    X = poly.fit_transform(np.array(X))
    x_test = poly.fit_transform(np.array(x_test))

    # split for validation testing
    x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.1)


    #X = normalize(X, axis=0)
    #x_test = normalize(x_test, axis=0)

    #Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
    #LassoLars(alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.2204460492503131e-16, copy_X=True, fit_path=True, positive=False)

    param_grid = [{'alpha':np.logspace(-4, 10, 1000)}]
    model = GridSearchCV(linear_model.LassoLars(), param_grid, cv=5, scoring=None, fit_params=None, n_jobs=-1, iid=True, refit=True, verbose=1, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

    print("Training...")
    model.fit(x_train, y_train)

    best_estimator = model.best_estimator_

    print 'Best score of Grid Search: ' + str(model.best_score_)
    print 'Best params of Grid Search: ' + str(model.best_params_)

    RMSE = mean_squared_error(y_validate, model.predict(x_validate))**0.5
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

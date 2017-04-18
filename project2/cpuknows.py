from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import normalize, PolynomialFeatures

def main():
    # set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")

    # load the data from the CSV files
    train_data = pd.read_csv('train.csv', header=0)
    test_data = pd.read_csv('test.csv', header=0)

    # transform the loaded CSV data into numpy arrays
    X = train_data.drop(['Id', 'y'], axis=1)
    Y = np.asarray(train_data['y'])
    ID = train_data['Id']

    t_id = test_data['Id']
    x_test = test_data.drop('Id', axis=1)

    # transform data
    #poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)
    #X = poly.fit_transform(np.array(X))
    #x_test = poly.fit_transform(np.array(x_test))

    # split for validation testing
    Xtrain, x_validate, Ytrain, y_validate = train_test_split(X, Y, test_size=0.1)

    rf_param_grid = {'n_estimators': [10, 30, 100, 300, 1000]}
    boost_param_grid = {'n_estimators': [10, 30, 100, 300, 1000],
                        'max_depth': [2, 3, 4, 5],
                        'min_samples_leaf': [1, 2, 3]}
    ada_param_grid = {'n_estimators': [10, 30, 100, 300, 1000],
                      'learning_rate': [0.1, 0.3, 1.0, 3.0]}



    print("Training...")
    rf_est = RandomForestClassifier()
    rf_gs_cv = GridSearchCV(rf_est, rf_param_grid).fit(Xtrain, Ytrain)
    print(rf_gs_cv.best_score_, rf_gs_cv.best_params_)
    print('\n')

    boost_est = GradientBoostingClassifier()
    boost_gs_cv = GridSearchCV(boost_est, boost_param_grid).fit(Xtrain, Ytrain)
    print(boost_gs_cv.best_score_, boost_gs_cv.best_params_)
    print('\n')

    ada_est = AdaBoostClassifier()
    ada_gs_cv = GridSearchCV(ada_est, ada_param_grid).fit(Xtrain, Ytrain)
    print(ada_gs_cv.best_score_, ada_gs_cv.best_params_)
    print('\n')

if __name__ == '__main__':
    main()

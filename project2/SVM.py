import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, PolynomialFeatures
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA


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
    # scale to unit variance
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    x_test = scaler.transform(x_test)
    # now take the 2 feature dimension which have most impact on class label
    pca = PCA(n_components=2)
    pca.fit(x_test)
    pca.fit(X)

    # split for validation testing
    x_train, x_validate, y_train, y_validate = train_test_split(X, Y)

    param_grid = [{'C':np.logspace(-1, 5, 20), 'kernel': ['rbf']}]
    model = GridSearchCV(svm.SVC(max_iter=1e8), param_grid, cv=10, scoring=None, fit_params=None, n_jobs=-1, iid=False, refit=True, verbose=1, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)

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

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import normalize, PolynomialFeatures
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

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

    # scale to unit variance
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    x_test = scaler.transform(x_test)
    # now take the 2 feature dimension which have most impact on class label
    pca = PCA(n_components=2)
    pca.fit(x_test)
    pca.fit(X)

    # split for validation testing
    x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.1)

    param_grid = {'n_estimators':[100,200,300,400,500,600,700],
                'criterion':["entropy"],
                'max_features':["sqrt"],
                    'min_samples_split':[2,3,4,5]
                    }

    model = GridSearchCV(ExtraTreesClassifier(), param_grid, cv=12, scoring=None, fit_params=None, n_jobs=-1, iid=False, refit=True, verbose=1, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)
    # model = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=3, random_state=0, n_jobs = 32)
    print("Training...")
    # model is trained on the training data extracted from csv
    model.fit(x_train, y_train)
    print 'Best score of Grid Search: ' + str(model.best_score_)
    print 'Best params of Grid Search: ' + str(model.best_params_)



    # print mean squared error as first estimate
    acc = accuracy_score(y_validate, model.predict(x_validate))
    print "Categorisation accuracy:"
    print acc

    print("Predicting...")

    # assemble data for csv creation
    y_prediction = model.predict(x_test)
    results_df = pd.DataFrame(data={'y':y_prediction})
    joined = pd.DataFrame(t_id).join(results_df)

    # save data in csv format for submission
    print("Writing predictions to predictions.csv")
    joined.to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()

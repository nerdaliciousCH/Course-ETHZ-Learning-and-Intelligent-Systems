import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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

    # split for validation testing
    x_train, x_validate, y_train, y_validate = train_test_split(X, Y)

    # params for gridsearch:
    # n_estimators=10,
    # min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    # max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, 
    # random_state=None, verbose=0, warm_start=False, 
    # class_weight=None)

    param_grid = [{'n_estimators':map(int, np.linspace(1, 1000, 100)),'criterion':['gini','entropy'],'max_features':['log2','sqrt'],'class_weight':['balanced_subsample','balanced']}]
    model = GridSearchCV(RandomForestClassifier(), param_grid, cv=20, scoring=None, fit_params=None, n_jobs=-1, iid=False, refit=True, verbose=1, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)


    print("Training...")

    # model is trained on the training data extracted from csv
    #model = GridSearchCV(models, parameters)
    model.fit(x_train, y_train)

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

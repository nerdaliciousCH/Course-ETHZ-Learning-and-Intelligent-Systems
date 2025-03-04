import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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

    model = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, 
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
        min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, 
        verbose=0, warm_start=False, class_weight=None)


    print("Training...")

    # model is trained on the training data extracted from csv
    #model = GridSearchCV(models, parameters)
    model.fit(x_train, y_train)

    # print mean squared error as first estimate
    RMSE = mean_squared_error(y_validate, model.predict(x_validate))**0.5
    print "Root median square error:"
    print RMSE

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


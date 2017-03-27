import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


def main():

    # set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    
    # load the data from the CSV files
    train_data = pd.read_csv('train.csv', header=0)
    test_data = pd.read_csv('test.csv', header=0)

    # transform the loaded CSV data into numpy arrays
    # X,Y is the training data
    # x_test the data to make a prediction on
    X = train_data.drop(['Id', 'y'], axis=1)
    Y = np.asarray(train_data['y'])

    # create a base classifier used to evaluate a subset of attributes
    model = LogisticRegression()

    # create the RFE model and select 3 attributes
    rfe = RFE(model, 3)
    rfe = rfe.fit(X, Y)

    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)

    # fit an Extra Trees model to the data
    model2 = ExtraTreesClassifier()
    model2.fit(X,Y)

    # display the relative importance of each attribute
    print(model2.feature_importances_)

if __name__ == '__main__':
    main()
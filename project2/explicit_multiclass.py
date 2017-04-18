import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

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

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    x_test = scaler.transform(x_test)


    x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.1)

    # was shitty
    #model = OneVsOneClassifier(svm.SVC(kernel='linear'),n_jobs=-1)


    # try all & use voting 
    clf1 = KNeighborsClassifier(n_neighbors=5,weights='distance')
    clf3 = svm.SVC(C=0.0011119677311206978, kernel='poly', degree=2, probability=True)
    #clf4 = svm.SVC(C=1, gamma=1, kernel='rbf', probability=True)
    clf6 = DecisionTreeClassifier(criterion='gini', max_depth=5)
    clf7 = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=2)
    clf8 = AdaBoostClassifier(n_estimators=100)
    clf9 = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    #clf9 = GaussianNB()
    clf4 = GaussianMixture(covariance_type='tied')


    model = VotingClassifier(estimators=[('k_n',clf1), ('svc1',clf3), ('svc2',clf4), ('dt', clf6), ('rf', clf7), ('abg', clf8), ('gpc',clf9)], voting = 'hard')


    print("Training...")
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
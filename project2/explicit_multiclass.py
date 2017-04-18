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


    x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.1)

    # was shitty
    #model = OneVsOneClassifier(svm.SVC(kernel='linear'),n_jobs=-1)


    # try all & use voting 
    clf1 = KNeighborsClassifier(n_neighbors=5,weights='distance')
    clf2 = svm.SVC(kernel="linear", C=1, gamma=1)
    clf3 = svm.SVC(C=1, gamma=1, kernel='poly', degree=3)
    clf4 = svm.SVC(C=1, gamma=1, kernel='rbf')
    clf5 = svm.SVC(C=1, gamma=1, kernel='poly', degree=2)
    clf6 = DecisionTreeClassifier(criterion='gini', max_depth=5)
    clf7 = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=2)
    clf8 = AdaBoostClassifier(n_estimators=100)
    clf9 = GaussianNB()
    clf10 = GaussianMixture(covariance_type='tied', init_params='wc', n_iter=20)


    model = VotingClassifier(estimators=[('k_n',clf1), ('svc1',clf2), ('svc2',clf3), 
    									 ('svc3',clf4), ('svc4',clf5), ('dt',clf6), 
    									 ('rt',clf7), ('AdaBoost',clf8), ('gnb',clf9), 
    									 ('gmm', clf10)], voting = 'soft')


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
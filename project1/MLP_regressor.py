import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


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

    # feature scaling for multi-layer perceptron
    #scaler = StandardScaler()
    #scaler.fit(x_train)  
    #x_train = scaler.transform(x_train)
    #x_test = scaler.transform(x_test)    

    # set up neural network and its arguments
    #parameters = {'alpha': (10.0 ** -np.arange(1, 7))}
    model = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='sgd', 
        alpha=0.01755, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, 
        power_t=0.5, max_iter=300, shuffle=True, random_state=None, tol=0.00001, verbose=False, 
        warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

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


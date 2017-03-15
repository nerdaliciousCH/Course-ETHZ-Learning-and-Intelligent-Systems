import pandas as pd
import numpy as np
from sklearn import linear_model


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    # Load the data from the CSV files
    train_data = pd.read_csv('train.csv', header=0)
    test_data = pd.read_csv('test.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = np.asarray(train_data['y'])
    train_id = train_data['Id']
    X = train_data.drop(['Id', 'y'], axis=1)
    t_id = test_data['Id']
    x_test = test_data.drop('Id', axis=1)

    # This is your model that will learn to predict
    #model = linear_model.LinearRegression(n_jobs=-1)
    alphas = np.logspace(-5, 10, 2)
    model = linear_model.RidgeCV(alphas, cv=10)

    print("Training...")
    # Your model is trained on the numerai_training_data
    model.fit(X, Y)

    print("Predicting...")
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict(x_test)
    #y_prediction = [np.average(x) for x in np.asarray(X)]
    results = y_prediction
    results_df = pd.DataFrame(data={'y':results})
    joined = pd.DataFrame(t_id).join(results_df)

    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()

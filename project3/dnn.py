import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from sklearn.metrics import accuracy_score
import sklearn.model_selection as sk


def main():

    # read input data from hd5 format
    train = pd.read_hdf("train.h5", "train")
    predict = pd.read_hdf("test.h5", "test")

    # create matrices from input
    Y = train.iloc[0:,0]
    X = train.iloc[0:,1:]
    X_predict = predict.iloc[0:,0:]
    t_id = predict.index.values

    # type conversion
    Y = np.array(Y.values).astype('int')
    X = np.array(X.values).astype('double')
    X_predict = np.array(X_predict.values).astype('double')

    X_train, X_test, Y_train, Y_test = sk.train_test_split(X,Y,test_size=0.05, random_state=42)

    #logging.getLogger().setLevel(logging.INFO)
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)


    # train
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=100)]

    # Train
    # tanh and softsign are the best activation functions so far
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[256,512,256,128],
                                                optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                enable_centered_bias=True,
                                                activation_fn=tf.nn.tanh,
                                                n_classes=5)
    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(X_train)
        y = tf.constant(Y_train)
        return x, y

    def get_test_inputs():
        x = tf.constant(X_test)
        y = tf.constant(Y_test)
        return x, y

    # Train a.k.a. Fit
    print("Training...")
    classifier.fit(input_fn=get_train_inputs, steps=1000)

    ## following line takes forever or doesn't halt... I don't know why
    # accuracy_score = classifier.evaluate(input_fn=get_test_inputs)["accuracy"]

    accuracy = accuracy_score(Y_test, list(classifier.predict(X_test)))
    print('Accuracy: {0:f}'.format(accuracy))

    # make the prediction
    print("Making predictions...")
    predictions = list(classifier.predict(X_predict))

    results_df = pd.DataFrame(data={'y':predictions})
    joined = pd.DataFrame({'Id':t_id}).join(results_df)

    # Save data in csv format for submission
    print("Writing predictions to predictions.csv")
    joined.to_csv("predictions.csv", index=False)

if __name__ == '__main__':
    main()

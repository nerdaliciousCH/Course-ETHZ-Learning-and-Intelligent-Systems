import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

def main():
    train = pd.read_hdf("train.h5", "train")
    test = pd.read_hdf("test.h5", "test")

    Y=train.iloc[0:,0]
    X=train.iloc[0:,1:]
    X_t=test.iloc[0:,0:]
    t_id=test.index.values

    Y=np.array(Y.values).astype('int')
    X=np.array(X.values).astype('double')
    X_t=np.array(X_t.values).astype('double')

    # Train
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=100)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20],
                                              n_classes=5,
                                              model_dir="/tmp/model")
    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(X)
        y = tf.constant(Y)
        return x, y

    # Train a.k.a. Fit
    classifier.fit(input_fn=get_train_inputs, steps=1000)

    # accuracy_score = classifier.evaluate(input_fn=get_train_inputs)["accuracy"]
    # print('Accuracy: {0:f}'.format(accuracy_score))

    # Predict
    predictions = list(classifier.predict(X_t))


    results_df = pd.DataFrame(data={'y':predictions})
    joined = pd.DataFrame({'Id':t_id}).join(results_df)

    # Save data in csv format for submission
    print("Writing predictions to predictions.csv")
    joined.to_csv("predictions.csv", index=False)

if __name__ == '__main__':
    main()

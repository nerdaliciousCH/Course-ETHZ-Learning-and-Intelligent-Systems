import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def main():
    train = pd.read_hdf("train.h5", "train")
    test = pd.read_hdf("test.h5", "test")



    # print mean squared error as first estimate
    acc = accuracy_score(y_validate, model.predict(x_validate))
    print("Categorisation accuracy:")
    print(acc)

    print("Predicting...")
    # assemble data for csv creation
    y_prediction = model.predict(x_test)
    results_df = pd.DataFrame(data={'y':y_prediction})
    joined = pd.DataFrame(t_id).join(results_df)

    # save data in csv format for submission
    print("Writing predictions to predictions.csv")
    joined.to_csv("predictions.csv", index=False)

    acc = accuracy_score(y, y_pred)


if __name__ == '__main__':
    main()

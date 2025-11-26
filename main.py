import numpy as np
from matplotlib import pyplot
from numpy._typing import NDArray
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    MNISTmini_path = "MNISTmini"
    data = loadmat(MNISTmini_path)

    train: list[NDArray] = data["train_fea1"]
    train_labels: list[NDArray] = data["train_gnd1"]
    test: list[NDArray] = data["test_fea1"]
    test_labels: list[NDArray] = data["test_gnd1"]

    training_fr = []
    training_labels_fr = []
    testing_fr = []
    testing_labels_fr = []

    for X, y in zip(train, train_labels):
        y_val = y[0]
        if y_val == [10] or y_val == [6]:
            training_fr.append(X)
            training_labels_fr.append(y_val)

    for X, y in zip(test, test_labels):
        y_val = y[0]
        if y_val == [10] or y_val == [6]:
            testing_fr.append(X)
            testing_labels_fr.append(y_val)

    training_fr_np = np.array(training_fr)
    training_labels_fr_np = np.array(training_labels_fr)
    testing_fr_np = np.array(testing_fr)
    testing_labels_fr_np = np.array(testing_labels_fr)

    logistic_regression_model = LogisticRegression(penalty="l2", solver="liblinear")
    # logistic_regression_model = RandomForestClassifier()
    logistic_regression_model.fit(training_fr_np, training_labels_fr_np)
    y_predict = logistic_regression_model.predict(testing_fr_np)
    print(testing_fr_np.shape)
    print(testing_labels_fr_np.shape)
    print(len(testing_labels_fr))
    print(y_predict.shape)
    accuracy = accuracy_score(testing_labels_fr_np, y_predict)
    print(f"accuracy: {accuracy}")

    # sample = training_fr_np[1500]
    # print(sample)
    # pixels = sample.reshape(10, 10).T
    # plot = pyplot
    # plot.imshow(pixels, cmap="gray")
    # plot.show()
    # print(pixels)
    # print(training_fr_np.shape)


if __name__ == "__main__":
    main()

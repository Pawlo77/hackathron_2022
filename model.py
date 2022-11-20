import numpy as np
from sklearn.svm import SVC


def load(name):
    return np.genfromtxt(name, delimiter=";")


if __name__ == "__main__":
    X_train = load("X_train.csv")
    y_train = load("y_train.csv")
    X_test = load("X_test.csv")

    svc = SVC()
    svc.fit(X_train, y_train)

    print(svc.score(X_train, y_train))

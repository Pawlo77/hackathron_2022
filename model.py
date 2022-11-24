import os
import pickle
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from skopt import BayesSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def load(name):
    return pd.read_csv(os.path.join("data", name)).to_numpy()


def report_search(search, name):
    with open("model_search_report.txt", "a") as f:
        print(
            f"Found {name} with best validation score of {search.best_score_}", file=f
        )
        print(f"For params: {search.best_params_}", file=f)


def save_model(model, name):
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)


def perform_search(X, y, cv, name, params, estimator):
    search = BayesSearchCV(estimator=estimator, search_spaces=params, n_jobs=-1, cv=cv)
    search.fit(X, y)
    report_search(search, name)

    return search.best_estimator_, search.best_score_


def create_SVC(X, y, cv, name):
    params = {
        "C": (1, 100, "log-uniform"),
        "gamma": (1, 100, "log-uniform"),
        "degree": (1, 3),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
    }
    return perform_search(X, y, cv, name, params, SVC())


def create_RFC(X, y, cv, name):
    params = {
        "n_estimators": (1, 1000, "log-uniform"),
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": (3, 6),
    }
    return perform_search(X, y, cv, name, params, RandomForestClassifier())


def create_KNN(X, y, cv, name):
    params = {
        "n_neighbors": (1, 100, "log-uniform"),
        "weights": ["distance", "uniform"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": (20, 50),
        "p": [1, 2],
    }
    return perform_search(X, y, cv, name, params, KNeighborsClassifier())


def create_LR(X, y, cv, name):
    params = {
        "C": (1, 10, "log-uniform"),
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["lbfgs", "sag", "saga"],
    }
    return perform_search(X, y, cv, name, params, LogisticRegression(max_iter=400))


if __name__ == "__main__":
    with open("model_search_report.txt", "w") as f:  # clear previous scores
        pass

    y_train = load("y_train.csv").ravel()
    y_test = load("y_test.csv").ravel()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    models = [
        (None, None, create_SVC, "SVC"),
        (None, None, create_RFC, "RandomForestClassifier"),
        (None, None, create_KNN, "KNeighborsClassifier"),
        (None, None, create_LR, "LogisticRegression"),
    ]

    for num_attrs in [1000, 2000, 5000]:
        X_train = load(f"X_train_{num_attrs}.csv")
        X_test = load(f"X_test_{num_attrs}.csv")

        for i in range(len(models)):
            model, score = models[i][2](X_train, y_train, cv, models[i][3])

            if models[i][0] is None or score > models[i][1]:
                models[i][0] = model
                models[i][1] = score

    with open("models_search_report.txt", "a") as f:
        for i in range(len(models)):
            model, name = models[i][0], models[i][3]
            test_scores = model.predict_proba(X_test)
            y_pred = np.argmax(test_scores, axis=0)

            print(f"Scores on training set for model {name}", file=f)
            for score in (f1_score, precision_score, recall_score, accuracy_score):
                print(f"\r{str(score)} - {score(y_test, y_pred)}")

            save_model(model, name)

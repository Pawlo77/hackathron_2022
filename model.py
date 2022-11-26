import os
import pickle
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def load(name):
    return pd.read_csv(os.path.join("data", name)).to_numpy()


def report_search(search, name):
    with open("model_search_report.txt", "a") as f:
        print(
            f"Found {name} with best validation score of {search.best_score_}", file=f
        )
        print(f"For params: {search.best_params_}", file=f)
    print("REPORTED")


def save_model(model, name):
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)


def perform_search(X, y, cv, name, params, estimator):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        n_jobs=-1,
        cv=cv,
        verbose=1,
    )
    search.fit(X, y)
    report_search(search, name)

    return search.best_estimator_, search.best_score_


def create_SVC(X, y, cv, name):
    params = {
        "C": [0.2, 0.5, 1, 2, 5],
        "gamma": ["auto", "scale"],
        "degree": [2, 3, 5],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
    }
    return perform_search(X, y, cv, name, params, SVC())


def create_RFC(X, y, cv, name):
    params = {
        "n_estimators": (
            100,
            250,
            500,
            800,
            1000,
        ),
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [3, 5, 8, 10],
    }
    return perform_search(X, y, cv, name, params, RandomForestClassifier())


def create_KNN(X, y, cv, name):
    params = {
        "n_neighbors": [5, 10, 20, 50],
        "weights": ["distance", "uniform"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 50],
        "p": [1, 2],
    }
    return perform_search(X, y, cv, name, params, KNeighborsClassifier())


def create_LR(X, y, cv, name):
    params = {
        "C": [0.2, 0.5, 1, 2, 5],
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["lbfgs", "sag", "saga"],
    }
    return perform_search(X, y, cv, name, params, LogisticRegression(max_iter=400))


def insert_model(model, score, num_attrs, name, best_models, k=3):
    if len(best_models) < k:
        best_models.append((model, score, num_attrs, name))
    else:
        for i in range(len(best_models)):
            if best_models[i][1] < score:
                best_models.insert(i, (model, score, num_attrs, name))
        return best_models[:k]


if __name__ == "__main__":
    with open("model_search_report.txt", "w") as f:  # clear previous scores
        pass

    y_train = load("y_train.csv").ravel()
    y_test = load("y_test.csv").ravel()
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    models = [
        (create_KNN, "KNeighborsClassifier"),
        (create_RFC, "RandomForestClassifier"),
        (create_LR, "LogisticRegression"),
        (create_SVC, "SVC"),
    ]
    best_models = []

    for num_attrs in [1000, 2000, 5000]:
        X_train = load(f"X_train_{num_attrs}.csv")

        for i in range(len(models)):
            create, name = models[i]
            model, score = create(X_train, y_train, 5, name)
            insert_model(model, score, num_attrs, name, best_models)

    with open("models_search_report.txt", "a") as f:
        for model, score, num_attrs, name in range(len(models)):
            X_test = load(f"X_test_{num_attrs}.csv")

            test_scores = model.predict_proba(X_test)
            y_pred = np.argmax(test_scores, axis=0)

            print(f"Scores on training set for model {name}_{num_attrs}", file=f)
            for score in (f1_score, precision_score, recall_score, accuracy_score):
                print(f"\r{str(score)} - {score(y_test, y_pred)}", file=f)

            save_model(model, f"{name}_{num_attrs}")

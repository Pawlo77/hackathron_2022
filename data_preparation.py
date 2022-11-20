import pickle
import os
import pandas as pd
from tools import XPipeline, FinalXPipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


def save_data(data, name):
    pd.DataFrame(data).to_csv(os.path.join("data", name), index=False)


def get_pipeline_file(name):
    return os.path.join("pipelines", name)


if __name__ == "__main__":
    X_train = pd.read_csv(os.path.join("data", "X_train_pure.csv"))
    y_train = (
        pd.read_csv(os.path.join("data", "y_train.csv"))["target"]
        .to_numpy()
        .reshape(-1, 1)
        .ravel()
    )

    X_test = pd.read_csv(os.path.join("nlp-getting-started", "test.csv"))
    y_test = (
        pd.read_csv(os.path.join("nlp-getting-started", "sample_submission.csv"))[
            "target"
        ]
        .to_numpy()
        .reshape(-1, 1)
        .ravel()
    )

    with open(get_pipeline_file("x_pipeline_pure.pkl"), "rb") as file:
        pure_pipeline = pickle.load(file)

    not_first_time = False
    for n_features_to_select in [1000, 2000, 5000]:
        print(f"Starting for {n_features_to_select}.")

        rcf = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            criterion="entropy",
            max_features=n_features_to_select // 20,
            n_jobs=-1,
        )
        selector = RFE(
            rcf,
            n_features_to_select=n_features_to_select,
            step=n_features_to_select // 20,
        )

        X_train_cur = selector.fit_transform(X_train.to_numpy(), y_train)
        print("Selector fitted.")

        save_data(X_train_cur, f"X_train_{n_features_to_select}.csv")
        del X_train_cur
        print(f"Generated: X_train_{n_features_to_select}.csv")

        x_pipeline = FinalXPipeline(n_features_to_select, pure_pipeline, selector)

        with open(
            get_pipeline_file(f"x_pipeline_{n_features_to_select}.pkl"), "wb"
        ) as file:
            pickle.dump(x_pipeline, file)
        print(f"Generated: x_pipeline_{n_features_to_select}.pkl")

        X_test_cur = x_pipeline.transform(X_test, hot_start=True)
        save_data(X_test_cur, f"X_test_{n_features_to_select}.csv")
        print(f"Generated: X_test_{n_features_to_select}.csv")

        if not not_first_time:
            save_data(y_train, f"y_train.csv")
            save_data(y_test, f"y_test.csv")

        not_first_time = True

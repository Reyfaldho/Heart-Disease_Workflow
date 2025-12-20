import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--target_col", type=str, default="target")
    p.add_argument("--experiment_name", type=str, default="CI_Retraining_HeartDisease")
    args = p.parse_args()

    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=[args.target_col, "num_original"], errors="ignore").astype("float64")
    y = df[args.target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("f1", f1_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

    mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()

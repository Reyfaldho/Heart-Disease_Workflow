import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="https://raw.githubusercontent.com/Reyfaldho/Eksperimen_SML_Reyfaldho-Alfarazel/refs/heads/main/Preprocessing/dataset_HeartDisease_membangun_sistem_machine_learning_preprocessing.csv")
    p.add_argument("--target_col", type=str, default="target")
    p.add_argument("--experiment_name", type=str, default="CI_Retraining_HeartDisease")
    p.add_argument("--tracking_uri", type=str, default="sqlite:///mlflow.db")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Kolom target '{args.target_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    X = df.drop(columns=[args.target_col, "num_original"], errors="ignore").astype("float64")
    y = df[args.target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="rf_ci_retrain"):
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("test_size", 0.2)

        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))

        mlflow.sklearn.log_model(model, name="model")


if __name__ == "__main__":
    main()



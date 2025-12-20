import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    args = parser.parse_args()

    mlflow.set_experiment("CI_Retraining_HeartDisease")

    df = pd.read_csv(args.data_path)

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model_type", "RandomForest")

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="HeartDiseaseModel"
    )

    print(f"Training finished. Accuracy: {acc}")

if __name__ == "__main__":
    main()

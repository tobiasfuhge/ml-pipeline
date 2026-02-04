import os
import json
import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--preprocessing-run-id", required=True)
    parser.add_argument("--model-type", required=True, choices=["lr", "rf", "gbm"])
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--lr-max-iter", type=int)
    parser.add_argument("--lr-tol", type=float)
    parser.add_argument("--rf-n-estimators", type=int)
    parser.add_argument("--rf-max-depth", type=int)
    parser.add_argument("--gbm-n-estimators", type=int)
    parser.add_argument("--gbm-learning-rate", type=float)
    return parser.parse_args()


def write_xcom(value):
    Path("/airflow/xcom").mkdir(parents=True, exist_ok=True)
    with open("/airflow/xcom/return.json", "w") as f:
        json.dump(value, f)


def load_preprocessed_data(run_id):
    client = mlflow.tracking.MlflowClient()
    tmp_dir = tempfile.mkdtemp()

    client.download_artifacts(run_id, "processed_data", tmp_dir)

    train_df = pd.read_csv(os.path.join(tmp_dir, "processed_data", "train.csv"))
    val_df = pd.read_csv(os.path.join(tmp_dir, "processed_data", "val.csv"))

    return (
        train_df.drop(columns=["target"]),
        val_df.drop(columns=["target"]),
        train_df["target"],
        val_df["target"],
    )


def evaluate(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        "f1_weighted": f1_score(y, preds, average="weighted"),
    }, preds


def log_confusion_matrix(y_true, y_pred, name):
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm)
    ax.set_title(name)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    mlflow.log_figure(fig, f"confusion_matrix_{name}.png")
    plt.close(fig)


def build_model(args):
    if args.model_type == "lr":
        return "LogisticRegression", LogisticRegression(
            max_iter=args.lr_max_iter,
            tol=args.lr_tol,
            solver="lbfgs",
        )

    if args.model_type == "rf":
        return "RandomForest", RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )

    if args.model_type == "gbm":
        return "GradientBoosting", GradientBoostingClassifier(
            n_estimators=args.gbm_n_estimators,
            learning_rate=args.gbm_learning_rate,
            random_state=args.random_state,
        )

    raise ValueError("Unknown model")


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    model_name, model = build_model(args)

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        mlflow.set_tag("pipeline_stage", "training")
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("preprocessing_run_id", args.preprocessing_run_id)

        X_train, X_val, y_train, y_val = load_preprocessed_data(args.preprocessing_run_id)

        model.fit(X_train, y_train)

        metrics, preds = evaluate(model, X_val, y_val)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=None,
                )

        log_confusion_matrix(y_val, preds, model_name)

        airflow_payload = {
            "run_id": run_id,
            "model": model_name,
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
        }

        write_xcom(airflow_payload)

        logger.info(f"Finished {model_name}: {airflow_payload}")


if __name__ == "__main__":
    main()

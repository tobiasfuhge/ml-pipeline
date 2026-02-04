import os
import json
import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return str(o)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--training-results", required=True)
    parser.add_argument("--preprocessing-run-id", required=True)
    parser.add_argument("--min-f1-macro", type=float, default=0.4)
    parser.add_argument("--min-precision-macro", type=float, default=0.3)
    parser.add_argument("--f1-drift-factor", type=float, default=0.8)
    return parser.parse_args()


def write_xcom(value):
    Path("/airflow/xcom").mkdir(parents=True, exist_ok=True)
    with open("/airflow/xcom/return.json", "w") as f:
        json.dump(value, f, default=json_default)


def load_val_data(preprocessing_run_id):
    client = mlflow.tracking.MlflowClient()
    tmp = tempfile.mkdtemp()

    client.download_artifacts(preprocessing_run_id, "processed_data", tmp)
    df = pd.read_csv(os.path.join(tmp, "processed_data", "val.csv"))

    return df.drop(columns=["target"]), df["target"].values


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    training_results = json.loads(args.training_results)

    best = max(training_results, key=lambda x: x["f1_macro"])

    logger.info(f"Best model: {best}")

    X_val, y_val = load_val_data(args.preprocessing_run_id)

    model_uri = f"runs:/{best['run_id']}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_val)

    val_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    val_precision = precision_score(y_val, y_pred, average="macro", zero_division=0)
    val_recall = recall_score(y_val, y_pred, average="macro", zero_division=0)

    gate_f1 = val_f1 >= args.min_f1_macro
    gate_precision = val_precision >= args.min_precision_macro
    gate_drift = val_f1 >= best["f1_macro"] * args.f1_drift_factor
    gate_sanity = bool(np.isin(y_pred, np.unique(y_val)).all())

    promote = gate_f1 and gate_precision and gate_drift and gate_sanity

    decision = {
        "best_model": best["model"],
        "best_run_id": best["run_id"],
        "metrics": {
            "val_f1_macro": val_f1,
            "val_precision_macro": val_precision,
            "val_recall_macro": val_recall,
        },
        "gates": {
            "f1": gate_f1,
            "precision": gate_precision,
            "drift": gate_drift,
            "sanity": gate_sanity,
        },
        "promote": promote,
    }

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(decision["metrics"])
        mlflow.log_param("best_model", best["model"])
        mlflow.set_tag("promote", str(promote).lower())

        os.makedirs("evaluation", exist_ok=True)
        json.dump(decision, open("evaluation/decision.json", "w"), indent=2)
        mlflow.log_artifacts("evaluation", artifact_path="evaluation")

    write_xcom(decision)

    logger.info(f"Evaluation finished: {decision}")


if __name__ == "__main__":
    main()

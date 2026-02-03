import os
import json
import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow.sklearn

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# JSON Helper
# ----------------------
def json_default(o):
    """Convert numpy types to native Python types for JSON serialization."""
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
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--preprocessing-run-id", required=True)
    parser.add_argument("--min-f1-macro", type=float, default=0.4)
    parser.add_argument("--min-precision-macro", type=float, default=0.3)
    parser.add_argument("--f1-drift-factor", type=float, default=0.8)
    return parser.parse_args()


def write_xcom(value):
    """Write a return file for Airflow XCom."""
    Path("/airflow/xcom").mkdir(parents=True, exist_ok=True)
    with open("/airflow/xcom/return.json", "w") as f:
        json.dump(value, f, default=json_default)


def load_best_model_and_data(training_run_id, preprocessing_run_id):
    """Load the best child run model and validation data."""
    if not training_run_id or training_run_id == "None":
        raise RuntimeError("Invalid training_run_id")
    if not preprocessing_run_id or preprocessing_run_id == "None":
        raise RuntimeError("Invalid preprocessing_run_id")

    client = MlflowClient()

    # --- Best Model ---
    training_run = client.get_run(training_run_id)
    experiment_id = training_run.info.experiment_id

    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{training_run_id}'",
    )

    if not child_runs:
        raise RuntimeError(f"No child runs found for training run {training_run_id}")

    best_run = max(child_runs, key=lambda r: r.data.metrics.get("f1_macro", 0.0))
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # --- Validation Data ---
    tmp_dir = tempfile.mkdtemp()
    client.download_artifacts(preprocessing_run_id, "processed_data", tmp_dir)
    val_path = os.path.join(tmp_dir, "processed_data", "val.csv")
    val_df = pd.read_csv(val_path)

    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"].values

    return best_run, model, X_val, y_val


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="evaluation") as run:
        eval_run_id = run.info.run_id
        logger.info(f"Evaluation run_id: {eval_run_id}")

        best_run, model, X_val, y_val = load_best_model_and_data(
            args.training_run_id, args.preprocessing_run_id
        )

        y_pred = model.predict(X_val)

        # --- Metrics ---
        val_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        val_precision = precision_score(y_val, y_pred, average="macro", zero_division=0)
        val_recall = recall_score(y_val, y_pred, average="macro", zero_division=0)
        train_f1 = best_run.data.metrics.get("f1_macro", val_f1)

        # --- Gates ---
        gate_f1 = bool(val_f1 >= args.min_f1_macro)
        gate_precision = bool(val_precision >= args.min_precision_macro)
        gate_drift = bool(val_f1 >= train_f1 * args.f1_drift_factor)
        valid_labels = np.unique(y_val)
        gate_sanity = bool(np.isin(y_pred, valid_labels).all())
        promote = gate_f1 and gate_precision and gate_drift and gate_sanity

        decision = {
            "best_model_run_id": best_run.info.run_id,
            "best_model_name": best_run.info.run_name,
            "metrics": {
                "val_f1_macro": float(val_f1),
                "val_precision_macro": float(val_precision),
                "val_recall_macro": float(val_recall),
                "train_f1_macro": float(train_f1),
            },
            "gates": {
                "f1_passed": gate_f1,
                "precision_passed": gate_precision,
                "drift_passed": gate_drift,
                "sanity_passed": gate_sanity,
            },
            "promote": promote,
        }

        # --- Logging ---
        mlflow.log_metrics({
            "val_f1_macro": val_f1,
            "val_precision_macro": val_precision,
            "val_recall_macro": val_recall,
        })
        mlflow.log_params({
            "best_model_run_id": best_run.info.run_id,
            "promote": promote,
            "num_clusters": len(valid_labels),
        })
        mlflow.set_tag("promote", str(promote).lower())
        failed = [k for k, v in decision["gates"].items() if not v]
        mlflow.set_tag("failed_gates", ",".join(failed) if failed else "none")

        # --- Save decision ---
        os.makedirs("evaluation", exist_ok=True)
        decision_path = os.path.join("evaluation", "decision.json")
        with open(decision_path, "w") as f:
            json.dump(decision, f, indent=2, default=json_default)
        mlflow.log_artifact(decision_path, artifact_path="evaluation")

        # --- Return for Airflow ---
        write_xcom({
            "evaluation_run_id": eval_run_id,
            "promote": promote,
        })

        logger.info(f"Evaluation finished. Promote = {promote}")


if __name__ == "__main__":
    main()

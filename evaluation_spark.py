import os
import json
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name-mlflow", required=True)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--bucket-name", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--min-f1-macro", type=float, default=0.4)
    p.add_argument("--min-precision-macro", type=float, default=0.3)
    p.add_argument("--f1-drift-factor", type=float, default=0.8)
    return p.parse_args()

def main():
    args = parse_args()

    spark = SparkSession.builder.appName("evaluation").getOrCreate()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    client = MlflowClient()

    with mlflow.start_run(run_name="evaluation"):

        # find best child run
        parent = client.get_run(args.training_run_id)
        exp = parent.info.experiment_id

        children = client.search_runs(
            experiment_ids=[exp],
            filter_string=f"tags.mlflow.parentRunId = '{args.training_run_id}'"
        )

        best = max(children, key=lambda r: r.data.metrics.get("f1_macro",0))

        model_uri = f"runs:/{best.info.run_id}/model"
        model = mlflow.spark.load_model(model_uri)

        base = f"s3a://{args.bucket_name}/{args.output_path}"
        val = spark.read.parquet(base + "/val")

        preds = model.transform(val)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

        val_f1 = evaluator.setMetricName("f1").evaluate(preds)
        val_precision = evaluator.setMetricName("weightedPrecision").evaluate(preds)
        val_recall = evaluator.setMetricName("weightedRecall").evaluate(preds)

        train_f1 = best.data.metrics.get("f1_macro", val_f1)

        gate_f1 = val_f1 >= args.min_f1_macro
        gate_prec = val_precision >= args.min_precision_macro
        gate_drift = val_f1 >= train_f1 * args.f1_drift_factor

        promote = all([gate_f1, gate_prec, gate_drift])

        decision = {
            "best_model_run_id": best.info.run_id,
            "best_model_name": best.info.run_name,
            "metrics": {
                "val_f1_macro": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall
            },
            "gates": {
                "f1_passed": gate_f1,
                "precision_passed": gate_prec,
                "drift_passed": gate_drift
            },
            "promote": promote
        }

        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/decision.json","w") as f:
            json.dump(decision,f,indent=2)

        mlflow.log_metrics({
            "val_f1_macro": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall
        })

        mlflow.log_param("promote", promote)
        mlflow.log_artifact("evaluation/decision.json", artifact_path="evaluation")

        print(promote)

if __name__ == "__main__":
    main()

import os
import argparse
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name-mlflow", required=True)
    p.add_argument("--preprocessing-run-id", required=True)
    p.add_argument("--bucket-name", required=True)
    p.add_argument("--output-path", required=True)
    return p.parse_args()

def main():
    args = parse_args()

    spark = SparkSession.builder.appName("training").getOrCreate()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="training") as parent:

        mlflow.set_tag("pipeline_stage", "training")
        mlflow.set_tag("preprocessing_run_id", args.preprocessing_run_id)

        base = f"s3a://{args.bucket_name}/{args.output_path}"

        train = spark.read.parquet(base + "/train")
        val = spark.read.parquet(base + "/val")

        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )

        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )

        models = {
            "logreg": LogisticRegression(featuresCol="scaled_features", labelCol="label", maxIter=50),
            "rf": RandomForestClassifier(featuresCol="scaled_features", labelCol="label", numTrees=200, maxDepth=10),
            "gbt": GBTClassifier(featuresCol="scaled_features", labelCol="label", maxIter=50)
        }

        best_f1 = 0
        best_name = None
        best_model = None

        for name, model in models.items():

            with mlflow.start_run(run_name=name, nested=True):

                fitted = model.fit(train)
                preds = fitted.transform(val)

                f1 = evaluator_f1.evaluate(preds)
                acc = evaluator_acc.evaluate(preds)

                mlflow.log_metrics({
                    "f1_macro": f1,
                    "accuracy": acc
                })

                # Confusion matrix
                cm = preds.groupBy("label", "prediction").count()
                cm_path = f"confusion_{name}"
                cm.coalesce(1).write.mode("overwrite").json(cm_path)
                mlflow.log_artifacts(cm_path)

                mlflow.spark.log_model(fitted, "model")

                if f1 > best_f1:
                    best_f1 = f1
                    best_name = name
                    best_model = fitted

        mlflow.log_metric("best_f1_macro", best_f1)
        mlflow.log_param("winning_model", best_name)

        print(best_name)

if __name__ == "__main__":
    main()

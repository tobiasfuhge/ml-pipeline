import os
import json
import argparse
import logging
from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np
import mlflow
import boto3
from botocore.client import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
    "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Dt_Customer", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
    "AcceptedCmp1", "AcceptedCmp2", "Complain", "Response",
    "umap_cluster", "month_name", "weekday_name"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--filename", required=True)
    return parser.parse_args()


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def load_data_from_s3(bucket_name, file_name):
    s3 = boto3.resource(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    obj = s3.Bucket(bucket_name).Object(file_name).get()
    return pd.read_csv(BytesIO(obj["Body"].read()))


def write_xcom(value):
    Path("/airflow/xcom").mkdir(parents=True, exist_ok=True)
    with open("/airflow/xcom/return.json", "w") as f:
        json.dump(value, f)


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="eda_customer_data") as run:
        run_id = run.info.run_id

        df = load_data_from_s3(args.bucket_name, args.filename)

        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise RuntimeError(f"Missing columns: {missing_cols}")

        df = df.dropna().drop_duplicates()

        missing_ratio = float(df.isnull().sum().sum()) / (df.shape[0] * df.shape[1])

        metrics = {
            "num_rows": int(len(df)),
            "num_columns": int(df.shape[1]),
            "missing_ratio": missing_ratio,
        }

        if "Income" in df.columns:
            metrics["income_mean"] = float(df["Income"].mean())

        if "Recency" in df.columns:
            metrics["recency_mean"] = float(df["Recency"].mean())

        if "umap_cluster" in df.columns:
            metrics["num_clusters"] = int(df["umap_cluster"].nunique())

        mlflow.log_metrics(metrics)

        # MLflow artifact
        os.makedirs("eda", exist_ok=True)
        json.dump(metrics, open("eda/metrics.json", "w"), indent=2)
        mlflow.log_artifacts("eda", artifact_path="eda")

        # 🚀 AIRFLOW PAYLOAD
        airflow_payload = {
            "run_id": run_id,
            **metrics
        }

        write_xcom(airflow_payload)

        logger.info(f"EDA finished: {airflow_payload}")


if __name__ == "__main__":
    main()

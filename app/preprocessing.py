import os
import json
import argparse
import logging
from pathlib import Path
from io import BytesIO

import boto3
from botocore.client import Config
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_xcom(value):
    Path("/airflow/xcom").mkdir(parents=True, exist_ok=True)
    with open("/airflow/xcom/return.json", "w") as f:
        json.dump(value, f)


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


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="preprocessing") as run:
        run_id = run.info.run_id

        df = load_data_from_s3(args.bucket_name, args.filename)

        drop_cols = [
            "month_name", "weekday_name", "Marital_Status",
            "Education", "Dt_Customer", "Year_Birth"
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        df = df.dropna().drop_duplicates()

        mnt_cols = [
            "MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds"
        ]
        df["TotalSpend"] = df[mnt_cols].sum(axis=1)

        cmp_cols = [
            "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
            "AcceptedCmp4", "AcceptedCmp5"
        ]
        df["TotalAcceptedCmp"] = df[cmp_cols].sum(axis=1)

        df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
        df["SpendPerWebVisit"] = df["TotalSpend"] / df["NumWebVisitsMonth"].replace(0, 1)

        target_col = "umap_cluster"
        if target_col not in df.columns:
            raise RuntimeError("Target column umap_cluster missing")

        y = df[target_col]

        numerical_cols = [
            "Recency", "NumDealsPurchases", "NumWebPurchases",
            "NumCatalogPurchases", "NumStorePurchases",
            "NumWebVisitsMonth", "Income", "TotalSpend",
            "TotalAcceptedCmp", "TotalChildren", "SpendPerWebVisit"
        ]

        categorical_cols = ["Complain", "Response"]

        X = df[categorical_cols + numerical_cols].copy()

        for c in categorical_cols:
            X[c] = X[c].astype("string").fillna("missing")

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ])

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )

        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        os.makedirs(args.output_path, exist_ok=True)

        train_path = os.path.join(args.output_path, "train.csv")
        val_path = os.path.join(args.output_path, "val.csv")

        pd.DataFrame(X_train_proc).assign(target=y_train.values)\
            .to_csv(train_path, index=False)

        pd.DataFrame(X_val_proc).assign(target=y_val.values)\
            .to_csv(val_path, index=False)

        num_features = X_train_proc.shape[1]

        mlflow.log_params({
            "target": target_col,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "num_numeric": len(numerical_cols),
            "num_categorical": len(categorical_cols),
            "num_features": num_features,
        })

        mlflow.log_metrics({
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "num_clusters": int(y.nunique()),
        })

        mlflow.log_artifacts(args.output_path, artifact_path="processed_data")
        mlflow.sklearn.log_model(preprocessor, artifact_path="preprocessing_model")

        airflow_payload = {
            "run_id": run_id,
            "train_path": train_path,
            "val_path": val_path,
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "num_features": int(num_features),
            "num_clusters": int(y.nunique()),
        }

        write_xcom(airflow_payload)

        logger.info(f"Preprocessing finished: {airflow_payload}")


if __name__ == "__main__":
    main()

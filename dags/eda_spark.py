import os
import json
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when, avg, sum as spark_sum

EXPECTED_COLUMNS = [
    "Recency","MntWines","MntFruits","MntMeatProducts","MntFishProducts",
    "MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases",
    "NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth",
    "Year_Birth","Education","Marital_Status","Income","Kidhome","Teenhome",
    "Dt_Customer","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5",
    "AcceptedCmp1","AcceptedCmp2","Complain","Response",
    "umap_cluster","month_name","weekday_name"
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name-mlflow", required=True)
    p.add_argument("--bucket-name", required=True)
    p.add_argument("--filename", required=True)
    return p.parse_args()

def main():
    import mlflow
    args = parse_args()

    spark = SparkSession.builder.appName("eda").getOrCreate()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="eda_customer_data") as run:

        df = spark.read.option("header", True).option("inferSchema", True).csv(
            f"s3a://{args.bucket_name}/{args.filename}"
        )

        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing columns: {missing}")

        df = df.dropna().dropDuplicates()

        row_count = df.count()
        col_count = len(df.columns)

        summary = {
            "num_rows": row_count,
            "num_columns": col_count,
            "columns": df.columns
        }

        os.makedirs("eda", exist_ok=True)

        with open("eda/summary.json","w") as f:
            json.dump(summary,f,indent=2)

        # Missing ratio
        missing_total = df.select([
            count(when(col(c).isNull() | isnan(c), c)).alias(c)
            for c in df.columns
        ]).collect()[0].asDict()

        total_missing = sum(missing_total.values())
        total_cells = row_count * col_count

        metrics = {
            "num_rows": row_count,
            "num_columns": col_count,
            "missing_ratio": total_missing / total_cells
        }

        if "Income" in df.columns:
            stats = df.agg(
                avg("Income").alias("mean"),
                ).collect()[0]
            metrics["income_mean"] = stats["mean"]

        if "Recency" in df.columns:
            metrics["recency_mean"] = df.agg(avg("Recency")).first()[0]

        spending_cols = [
            "MntWines","MntFruits","MntMeatProducts",
            "MntFishProducts","MntSweetProducts","MntGoldProds"
        ]

        if all(c in df.columns for c in spending_cols):
            df = df.withColumn(
                "total_spent",
                sum(col(c) for c in spending_cols)
            )
            metrics["avg_total_spent"] = df.agg(avg("total_spent")).first()[0]

        if "umap_cluster" in df.columns:
            clusters = df.groupBy("umap_cluster").count()
            total = clusters.agg(sum("count")).first()[0]

            metrics["num_clusters"] = clusters.count()

            for r in clusters.collect():
                metrics[f"cluster_{r['umap_cluster']}_ratio"] = r["count"]/total

        mlflow.log_metrics(metrics)

        # Airflow XCom
        print(run.info.run_id)

if __name__ == "__main__":
    main()

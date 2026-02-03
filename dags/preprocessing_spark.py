import os
import argparse
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, max as spark_max
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    Imputer
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name-mlflow", required=True)
    p.add_argument("--bucket-name", required=True)
    p.add_argument("--filename", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    spark = SparkSession.builder.appName("preprocessing").getOrCreate()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="preprocessing") as run:

        df = spark.read.option("header", True).option("inferSchema", True).csv(
            f"s3a://{args.bucket_name}/{args.filename}"
        )

        # drop cols
        df = df.drop("month_name", "weekday_name")

        df = df.dropna().dropDuplicates()

        df = df.filter(~col("Marital_Status").isin("YOLO","Absurd"))

        # feature engineering
        max_date = df.select(spark_max("Dt_Customer")).first()[0]
        df = df.withColumn("Customer_Days", datediff(max_date, col("Dt_Customer")))

        target = "umap_cluster"

        numerical = [
            "Recency","MntWines","MntFruits","MntMeatProducts","MntFishProducts",
            "MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases",
            "NumCatalogPurchases","NumStorePurchases","NumWebVisitsMonth",
            "Income","Kidhome","Teenhome","Customer_Days","Year_Birth"
        ]

        categorical = [
            "Education","Marital_Status",
            "AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5",
            "Complain","Response"
        ]

        # Index categoricals
        indexers = [
            StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            for c in categorical
        ]

        encoder = OneHotEncoder(
            inputCols=[f"{c}_idx" for c in categorical],
            outputCols=[f"{c}_ohe" for c in categorical]
        )

        imputer = Imputer(
            inputCols=numerical,
            outputCols=[f"{c}_imp" for c in numerical]
        )

        assembler = VectorAssembler(
            inputCols=[f"{c}_imp" for c in numerical] + [f"{c}_ohe" for c in categorical],
            outputCol="features"
        )

        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features"
        )

        pipeline = Pipeline(stages=indexers + [encoder, imputer, assembler, scaler])

        model = pipeline.fit(df)
        processed = model.transform(df).select("scaled_features", col(target).alias("label"))

        train, val = processed.randomSplit([1-args.test_size, args.test_size], seed=args.random_state)

        output = f"s3a://{args.bucket_name}/{args.output_path}"

        train.write.mode("overwrite").parquet(output + "/train")
        val.write.mode("overwrite").parquet(output + "/val")

        mlflow.log_params({
            "test_size": args.test_size,
            "num_categorical": len(categorical),
            "num_numeric": len(numerical)
        })

        mlflow.log_metrics({
            "train_rows": train.count(),
            "val_rows": val.count()
        })

        mlflow.spark.log_model(model, "preprocessing_pipeline")

        print(run.info.run_id)

if __name__ == "__main__":
    main()

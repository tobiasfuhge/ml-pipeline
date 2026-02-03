from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

# --- Default DAG args ---
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --- DAG Definition ---
with DAG(
    dag_id="ml_pipeline_spark",
    default_args=default_args,
    description="ML pipeline with Spark (EDA, Preprocessing, Training, Evaluation)",
    schedule_interval=None,  # manuell triggern
    start_date=datetime(2026, 2, 3),
    catchup=False,
) as dag:

    # --- 1. EDA ---
    eda = SparkSubmitOperator(
        task_id="eda",
        application="/opt/spark/app/eda_spark.py",
        conn_id="spark_default",
        verbose=1,
        conf={
            "spark.kubernetes.container.image": "ghcr.io/einjit/ml-pipeline:1.0",
            "spark.kubernetes.namespace": "spark",
        },
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--bucket-name", "input-data",
            "--filename", "customer-segmentation.csv"
        ]
    )

    # --- 2. Preprocessing ---
    preprocess = SparkSubmitOperator(
        task_id="preprocess",
        application="/opt/spark/app/preprocessing_spark.py",
        conn_id="spark_default",
        verbose=1,
        conf={
            "spark.kubernetes.container.image": "ghcr.io/einjit/ml-pipeline:1.0",
            "spark.kubernetes.namespace": "spark",
        },
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--bucket-name", "input-data",
            "--filename", "customer-segmentation.csv",
            "--output-path", "data/processed/",
            "--test-size", "0.2",
            "--random-state", "42"
        ]
    )

    # --- 3. Training ---
    train = SparkSubmitOperator(
        task_id="train",
        application="/opt/spark/app/train_spark.py",
        conn_id="spark_default",
        verbose=1,
        conf={
            "spark.kubernetes.container.image": "ghcr.io/einjit/ml-pipeline:1.0",
            "spark.kubernetes.namespace": "spark",
        },
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--preprocessing-run-id", "/tmp/preprocessing_run_id.txt"
        ]
    )

    # --- 4. Evaluation ---
    evaluate = SparkSubmitOperator(
        task_id="evaluate",
        application="/opt/spark/app/evaluation_spark.py",
        conn_id="spark_default",
        verbose=1,
        conf={
            "spark.kubernetes.container.image": "ghcr.io/einjit/ml-pipeline:1.0",
            "spark.kubernetes.namespace": "spark",
        },
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--training-run-id", "/tmp/train_run_id.txt",
            "--preprocessing-run-id", "/tmp/preprocessing_run_id.txt",
            "--min-f1-macro", "0.5",
            "--min-precision-macro", "0.5",
            "--f1-drift-factor", "0.5"
        ]
    )

    # --- Task Dependencies ---
    eda >> preprocess >> train >> evaluate

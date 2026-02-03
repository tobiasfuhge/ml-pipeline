from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

# --- Basiskonfiguration für alle Spark-Tasks (angepasst für Standalone Cluster) ---
# Wir entfernen die irrelevanten Kubernetes-Konfigurationen.
base_spark_conf = {
    "spark.submit.deployMode": "cluster",
}

# --- Standard-DAG-Argumente ---
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

# --- DAG Definition ---
with DAG(
    dag_id="ml_pipeline_spark_manual_trigger",
    default_args=default_args,
    description="ML-Pipeline mit Spark, manuell triggerbar (EDA, Preprocessing, Training, Evaluation)",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["spark", "ml", "standalone"],
) as dag:

    # --- 1. EDA ---
    eda = SparkSubmitOperator(
        task_id="eda",
        application="{{ dag.folder }}/eda_spark.py",
        conn_id="spark_k8s_conn",  # Der Name ist ok, auch wenn es kein k8s-Mode mehr ist
        verbose=1,
        conf=base_spark_conf, # Verwendet die neue, saubere Konfiguration
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--bucket-name", "input-data",
            "--filename", "customer-segmentation.csv"
        ]
    )

    # --- 2. Preprocessing ---
    preprocess = SparkSubmitOperator(
        task_id="preprocess",
        application="{{ dag.folder }}/preprocessing_spark.py",
        conn_id="spark_k8s_conn",
        verbose=1,
        conf=base_spark_conf,
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--bucket-name", "input-data",
            "--filename", "customer-segmentation.csv",
            "--output-path", "data_spark/processed/",
            "--test-size", "0.2",
            "--random-state", "42"
        ],
        do_xcom_push=True,
    )

    # --- 3. Training ---
    train = SparkSubmitOperator(
        task_id="train",
        application="{{ dag.folder }}/train_spark.py",
        conn_id="spark_k8s_conn",
        verbose=1,
        conf=base_spark_conf,
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess') }}",
        ],
        do_xcom_push=True,
    )

    # --- 4. Evaluation ---
    evaluate = SparkSubmitOperator(
        task_id="evaluate",
        application="{{ dag.folder }}/evaluation_spark.py",
        conn_id="spark_k8s_conn",
        verbose=1,
        conf=base_spark_conf,
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--training-run-id", "{{ ti.xcom_pull(task_ids='train') }}",
            "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess') }}",
            "--min-f1-macro", "0.5",
            "--min-precision-macro", "0.5",
            "--f1-drift-factor", "0.5"
        ]
    )

    # --- Task-Abhängigkeiten festlegen ---
    eda >> preprocess >> train >> evaluate
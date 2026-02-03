from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

# --- Basiskonfiguration für alle Spark-Tasks ---
# Dies reduziert Code-Wiederholungen und macht den DAG übersichtlicher.
base_spark_conf = {
    "spark.submit.deployMode": "cluster",
    "spark.kubernetes.container.image": "ghcr.io/tobiasfuhge/ml-pipeline:1.0",
    "spark.kubernetes.namespace": "airflow",
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
    schedule=None,  # Ermöglicht manuelles Triggern über die UI
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["spark", "ml", "kubernetes"],
) as dag:

    # --- 1. EDA ---
    eda = SparkSubmitOperator(
        task_id="eda",
        application="/opt/spark/app/eda_spark.py",
        conn_id="spark_k8s_conn",  # Verwendet die in der UI erstellte Verbindung
        verbose=1,
        conf=base_spark_conf,
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--bucket-name", "input-data",
            "--filename", "customer-segmentation.csv"
        ]
    )

    # --- 2. Preprocessing ---
    # do_xcom_push=True sorgt dafür, dass die letzte Zeile der Standardausgabe
    # (z.B. die run_id) als XCom gespeichert wird.
    preprocess = SparkSubmitOperator(
        task_id="preprocess",
        application="/opt/spark/app/preprocessing_spark.py",
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
    # Hier holen wir die run_id vom 'preprocess'-Task über XCom.
    train = SparkSubmitOperator(
        task_id="train",
        application="/opt/spark/app/train_spark.py",
        conn_id="spark_k8s_conn",
        verbose=1,
        conf=base_spark_conf,
        application_args=[
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            # Jinja-Templating, um den XCom-Wert vom vorherigen Task zu holen
            "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess') }}",
        ],
        do_xcom_push=True,
    )

    # --- 4. Evaluation ---
    # Hier werden die run_ids von 'preprocess' und 'train' per XCom geholt.
    evaluate = SparkSubmitOperator(
        task_id="evaluate",
        application="/opt/spark/app/evaluation_spark.py",
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
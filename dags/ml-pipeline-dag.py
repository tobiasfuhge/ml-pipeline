from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.spark_kubernetes import SparkKubernetesOperator
from airflow.providers.cncf.kubernetes.sensors.spark_kubernetes import SparkKubernetesSensor
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_pipeline_spark_k8s",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 2, 3),
    catchup=False,
) as dag:

    # --- Hilfsfunktion für SparkApplication YAML ---
    def create_spark_app(name, file_path, args):
        return {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "SparkApplication",
            "metadata": {"name": name, "namespace": "spark"},
            "spec": {
                "type": "Python",
                "mode": "cluster",
                "image": "ghcr.io/tobiasfuhge/ml-pipeline:1.0",
                "pythonVersion": "3",
                "mainApplicationFile": f"local://{file_path}",
                "arguments": args,
                "sparkVersion": "3.5.0",
                "driver": {
                    "cores": 1,
                    "memory": "512m",
                    "serviceAccount": "spark-operator-spark", # Prüfe diesen Namen!
                },
                "executor": {
                    "cores": 1,
                    "instances": 1,
                    "memory": "512m",
                },
            },
        }

    # --- 1. EDA ---
    eda_submit = SparkKubernetesOperator(
        task_id="eda_submit",
        namespace="spark",
        template_spec=create_spark_app("ml-eda", "/opt/spark/app/eda_spark.py", [
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--bucket-name", "input-data",
            "--filename", "customer-segmentation.csv"
        ]),
    )
    
    eda_wait = SparkKubernetesSensor(
        task_id="eda_wait",
        namespace="spark",
        application_name="ml-eda",
    )

    # --- 2. Preprocessing ---
    preprocess_submit = SparkKubernetesOperator(
        task_id="preprocess_submit",
        namespace="spark",
        template_spec=create_spark_app("ml-preprocess", "/opt/spark/app/preprocessing_spark.py", [
            "--experiment-name-mlflow", "airflow-ml-pipeline",
            "--bucket-name", "input-data",
            "--filename", "customer-segmentation.csv",
            "--output-path", "data_spark/processed/"
        ]),
    )

    preprocess_wait = SparkKubernetesSensor(
        task_id="preprocess_wait",
        namespace="spark",
        application_name="ml-preprocess",
    )

    # --- Task Dependencies ---
    eda_submit >> eda_wait >> preprocess_submit >> preprocess_wait
    # (Die restlichen Tasks train/evaluate folgen dem gleichen Muster)
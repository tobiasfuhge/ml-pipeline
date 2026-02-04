from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from airflow.operators.short_circuit import ShortCircuitOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.timezone import utcnow
from airflow.models.param import Param

# ----------------------------
# Defaults
# ----------------------------

default_args = {
    "owner": "admin",
    "retries": 2,
}

# ----------------------------
# Constants
# ----------------------------

IMAGE = "ghcr.io/tobiasfuhge/data-eng:3.0"

# ----------------------------
# Secrets
# ----------------------------

aws_access_key = Secret("env", "AWS_ACCESS_KEY_ID", "minio", "root-user")
aws_secret_key = Secret("env", "AWS_SECRET_ACCESS_KEY", "minio", "root-password")
pg_user = Secret("env", "POSTGRES_USER", "postgres-admin", "postgres-user")
pg_password = Secret("env", "POSTGRES_PASSWORD", "postgres-admin", "postgres-password")

SECRETS = [aws_access_key, aws_secret_key, pg_user, pg_password]

# ----------------------------
# Shared ENV
# ----------------------------

COMMON_ENV = {
    "MLFLOW_S3_ENDPOINT_URL": "http://minio.data-storage.svc.cluster.local:9000",
    "MLFLOW_TRACKING_URI": "http://mlflow.mlops.svc.cluster.local:5000",
    "AWS_DEFAULT_REGION": "us-east-1",
    "POSTGRES_HOST": "postgres-postgresql.data-storage.svc.cluster.local",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "ds_db",
}

# ----------------------------
# Gate Function
# ----------------------------

def promotion_gate(ti):
    result = ti.xcom_pull(task_ids="ml.evaluate")
    return result["promote"]

# ----------------------------
# DAG
# ----------------------------

with DAG(
    dag_id="ml_pipeline_airflow_native",
    start_date=utcnow(),
    catchup=False,
    schedule=None,
    default_args=default_args,

    params={
        "experiment": Param("airflow-pipeline"),
        "bucket": Param("input-data"),
        "file": Param("customer-segmentation.csv"),
    }
) as dag:

    # =====================
    # ML PIPELINE
    # =====================

    with TaskGroup("ml") as ml:

        eda = KubernetesPodOperator(
            task_id="eda",
            image=IMAGE,
            cmds=["python", "eda.py"],
            arguments=[
                "--experiment-name-mlflow", "{{ params.experiment }}",
                "--bucket-name", "{{ params.bucket }}",
                "--filename", "{{ params.file }}"
            ],
            namespace="airflow",
            secrets=SECRETS,
            env_vars=COMMON_ENV,
            get_logs=True,
            on_finish_action="delete_succeeded_pod",
        )

        preprocess = KubernetesPodOperator(
            task_id="preprocess",
            image=IMAGE,
            cmds=["python", "preprocessing.py"],
            arguments=[
                "--experiment-name-mlflow", "{{ params.experiment }}",
                "--bucket-name", "{{ params.bucket }}",
                "--filename", "{{ params.file }}",
                "--output-path", "airflow-data/processed/",
                "--test-size", "0.2",
                "--random-state", "42",
            ],
            namespace="airflow",
            secrets=SECRETS,
            env_vars=COMMON_ENV,
            do_xcom_push=True,
        )

        train = KubernetesPodOperator(
            task_id="train",
            image=IMAGE,
            cmds=["python", "train.py"],
            arguments=[
                "--experiment-name-mlflow", "{{ params.experiment }}",
                "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='ml.preprocess') }}",
                "--random-state", "42",
                "--lr-max-iter", "100",
                "--lr-tol", "0.0001",
                "--rf-n-estimators", "100",
                "--rf-max-depth", "10",
                "--gbm-n-estimators", "100",
                "--gbm-learning-rate", "0.1",
            ],
            namespace="airflow",
            secrets=SECRETS,
            env_vars=COMMON_ENV,
            do_xcom_push=True,
        )

        evaluate = KubernetesPodOperator(
            task_id="evaluate",
            image=IMAGE,
            cmds=["python", "evaluation.py"],
            arguments=[
                "--experiment-name-mlflow", "{{ params.experiment }}",
                "--training-run-id", "{{ ti.xcom_pull(task_ids='ml.train') }}",
                "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='ml.preprocess') }}",
                "--min-f1-macro", "0.5",
                "--min-precision-macro", "0.5",
                "--f1-drift-factor", "0.5",
            ],
            namespace="airflow",
            secrets=SECRETS,
            env_vars=COMMON_ENV,
            do_xcom_push=True,
        )

        eda >> preprocess >> train >> evaluate

    # =====================
    # GATE
    # =====================

    gate = ShortCircuitOperator(
        task_id="promotion_gate",
        python_callable=promotion_gate,
    )

    # =====================
    # PROMOTION
    # =====================

    promote = KubernetesPodOperator(
        task_id="promote_model",
        image=IMAGE,
        cmds=["echo", "PROMOTING MODEL 🚀"],
        namespace="airflow",
    )

    ml >> gate >> promote

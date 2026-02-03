from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from airflow.utils.timezone import utcnow

default_args = {"owner": "admin"}

EXPERIMENT_NAME = "airflow-pipeline"
BUCKET_NAME = "input-data"
FILENAME = "customer-segmentation.csv"
OUTPUT_PATH = "airflow-data/processed/"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_F1 = 0.5
MIN_PRECISION = 0.5
F1_DRIFT = 0.5
LR_MAX_ITER = 100
LR_TOL = 0.0001
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
GBM_N_ESTIMATORS = 100
GBM_LEARNING_RATE = 0.1

# ----------------------------
# Secrets aus Kubernetes
# ----------------------------
aws_access_key = Secret(
    deploy_type='env',
    deploy_target='AWS_ACCESS_KEY_ID',
    secret='minio',
    key='root-user'
)
aws_secret_key = Secret(
    deploy_type='env',
    deploy_target='AWS_SECRET_ACCESS_KEY',
    secret='minio',
    key='root-password'
)
pg_user = Secret(
    deploy_type='env',
    deploy_target='POSTGRES_USER',
    secret='postgres-admin',
    key='postgres-user'
)
pg_password = Secret(
    deploy_type='env',
    deploy_target='POSTGRES_PASSWORD',
    secret='postgres-admin',
    key='postgres-password'
)

# ----------------------------
# DAG Definition
# ----------------------------
with DAG(
    dag_id="ml_pipeline_argo_1to1",
    default_args=default_args,
    schedule=None,
    start_date=utcnow(),
    catchup=False,
) as dag:

    # --------------------
    # 1. EDA Pod
    # --------------------
    eda = KubernetesPodOperator(
        task_id="eda",
        name="eda-pod",
        namespace="airflow",
        image="ghcr.io/tobiasfuhge/data-eng:1.0",
        cmds=["python", "eda.py"],
        arguments=[
            "--experiment-name-mlflow", EXPERIMENT_NAME,
            "--bucket-name", BUCKET_NAME,
            "--filename", FILENAME
        ],
        on_finish_action='keep_pod',
        get_logs=True,
        secrets=[aws_access_key, aws_secret_key, pg_user, pg_password],
        env_vars={
            "MLFLOW_S3_ENDPOINT_URL": "http://minio.data-storage.svc.cluster.local:9000",
            "MLFLOW_TRACKING_URI": "http://mlflow.mlops.svc.cluster.local:5000",
            "AWS_DEFAULT_REGION": "us-east-1",
            "POSTGRES_HOST": "postgres-postgresql.data-storage.svc.cluster.local",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "ds_db"
        }
    )

    # --------------------
    # 2. Preprocess Pod
    # --------------------
    preprocess = KubernetesPodOperator(
        task_id="preprocess",
        name="preprocess-pod",
        namespace="airflow",
        image="ghcr.io/tobiasfuhge/data-eng:1.0",
        cmds=["python", "preprocessing.py"],
        arguments=[
            "--experiment-name-mlflow", EXPERIMENT_NAME,
            "--bucket-name", BUCKET_NAME,
            "--filename", FILENAME,
            "--output-path", OUTPUT_PATH,
            "--test-size", str(TEST_SIZE),
            "--random-state", str(RANDOM_STATE)
        ],
        get_logs=True,
        on_finish_action='keep_pod',
        do_xcom_push=True,
        secrets=[aws_access_key, aws_secret_key, pg_user, pg_password],
        env_vars={
            "MLFLOW_S3_ENDPOINT_URL": "http://minio.data-storage.svc.cluster.local:9000",
            "MLFLOW_TRACKING_URI": "http://mlflow.mlops.svc.cluster.local:5000",
            "AWS_DEFAULT_REGION": "us-east-1",
            "POSTGRES_HOST": "postgres-postgresql.data-storage.svc.cluster.local",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "ds_db"
        }
    )

    # --------------------
    # 3. Train Pod
    # --------------------
    train = KubernetesPodOperator(
        task_id="train",
        name="train-pod",
        namespace="airflow",
        image="ghcr.io/tobiasfuhge/data-eng:1.0",
        cmds=["python", "train.py"],
        arguments=[
            "--experiment-name-mlflow", EXPERIMENT_NAME,
            "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess') }}",
            "--random-state", str(RANDOM_STATE),
            "--lr-max-iter", str(LR_MAX_ITER),
            "--lr-tol", str(LR_TOL),
            "--rf-n-estimators", str(RF_N_ESTIMATORS),
            "--rf-max-depth", str(RF_MAX_DEPTH),
            "--gbm-n-estimators", str(GBM_N_ESTIMATORS),
            "--gbm-learning-rate", str(GBM_LEARNING_RATE)
        ],
        get_logs=True,
        on_finish_action='keep_pod',
        do_xcom_push=True,
        secrets=[aws_access_key, aws_secret_key, pg_user, pg_password],
        env_vars={
            "MLFLOW_S3_ENDPOINT_URL": "http://minio.data-storage.svc.cluster.local:9000",
            "MLFLOW_TRACKING_URI": "http://mlflow.mlops.svc.cluster.local:5000",
            "AWS_DEFAULT_REGION": "us-east-1",
            "POSTGRES_HOST": "postgres-postgresql.data-storage.svc.cluster.local",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "ds_db"
        }
    )

    # --------------------
    # 4. Evaluate Pod
    # --------------------
    evaluate = KubernetesPodOperator(
        task_id="evaluate",
        name="evaluate-pod",
        namespace="airflow",
        image="ghcr.io/tobiasfuhge/data-eng:1.0",
        cmds=["python", "evaluation.py"],
        arguments=[
            "--experiment-name-mlflow", EXPERIMENT_NAME,
            "--training-run-id", "{{ ti.xcom_pull(task_ids='train') }}",
            "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess') }}",
            "--min-f1-macro", str(MIN_F1),
            "--min-precision-macro", str(MIN_PRECISION),
            "--f1-drift-factor", str(F1_DRIFT)
        ],
        get_logs=True,
        on_finish_action='keep_pod',
        secrets=[aws_access_key, aws_secret_key, pg_user, pg_password],
        env_vars={
            "MLFLOW_S3_ENDPOINT_URL": "http://minio.data-storage.svc.cluster.local:9000",
            "MLFLOW_TRACKING_URI": "http://mlflow.mlops.svc.cluster.local:5000",
            "AWS_DEFAULT_REGION": "us-east-1",
            "POSTGRES_HOST": "postgres-postgresql.data-storage.svc.cluster.local",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "ds_db"
        }
    )

    # --------------------
    # DAG Reihenfolge
    # --------------------
    eda >> preprocess >> train >> evaluate

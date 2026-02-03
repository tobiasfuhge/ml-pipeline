from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.utils.dates import days_ago

default_args = {"owner": "airflow"}

EXPERIMENT_NAME = "aiflow-pipeline"
BUCKET_NAME = "input-data"
FILENAME = "customer-segmentation.csv"
OUTPUT_PATH = "data-airflow/processed/"
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

with DAG(
    dag_id="simple_ml_pipeline",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
) as dag:

    # --------------------
    # 1. EDA Pod
    # --------------------
    eda = KubernetesPodOperator(
        task_id="eda",
        name="eda-pod",
        namespace="default",
        image="ghcr.io/einjit/data-eng:1.0",
        cmds=["python", "eda.py"],
        arguments=[
            "--experiment-name-mlflow", EXPERIMENT_NAME,
            "--bucket-name", BUCKET_NAME,
            "--filename", FILENAME
        ],
        get_logs=True,
        is_delete_operator_pod=True
    )

    # --------------------
    # 2. Preprocess Pod
    # --------------------
    preprocess = KubernetesPodOperator(
        task_id="preprocess",
        name="preprocess-pod",
        namespace="default",
        image="ghcr.io/einjit/data-eng:1.0",
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
        is_delete_operator_pod=True,
        do_xcom_push=True
    )

    # --------------------
    # 3. Train Pod
    # --------------------
    train = KubernetesPodOperator(
        task_id="train",
        name="train-pod",
        namespace="default",
        image="ghcr.io/einjit/data-eng:1.0",
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
        is_delete_operator_pod=True,
        do_xcom_push=True
    )

    # --------------------
    # 4. Evaluate Pod
    # --------------------
    evaluate = KubernetesPodOperator(
        task_id="evaluate",
        name="evaluate-pod",
        namespace="default",
        image="ghcr.io/einjit/data-eng:1.0",
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
        is_delete_operator_pod=True
    )

    # --------------------
    # DAG Reihenfolge
    # --------------------
    eda >> preprocess >> train >> evaluate

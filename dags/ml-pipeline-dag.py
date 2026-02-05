from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.timezone import utcnow
from airflow.models.param import Param



default_args = {
    "owner": "admin",
    "retries": 0,
}

IMAGE = "ghcr.io/tobiasfuhge/data-eng:10.0"


aws_access_key = Secret("env", "AWS_ACCESS_KEY_ID", "minio", "root-user")
aws_secret_key = Secret("env", "AWS_SECRET_ACCESS_KEY", "minio", "root-password")
pg_user = Secret("env", "POSTGRES_USER", "postgres-admin", "postgres-user")
pg_password = Secret("env", "POSTGRES_PASSWORD", "postgres-admin", "postgres-password")

SECRETS = [aws_access_key, aws_secret_key, pg_user, pg_password]

COMMON_ENV = {
    "MLFLOW_S3_ENDPOINT_URL": "http://minio.data-storage.svc.cluster.local:9000",
    "MLFLOW_TRACKING_URI": "http://mlflow.mlops.svc.cluster.local:5000",
    "AWS_DEFAULT_REGION": "us-east-1",
    "POSTGRES_HOST": "postgres-postgresql.data-storage.svc.cluster.local",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "ds_db",
}


def promotion_gate(ti):
    result = ti.xcom_pull(task_ids="evaluate")
    return result.get("promote", False)


with DAG(
    dag_id="ml_pipeline_parallel_training",
    start_date=utcnow(),
    catchup=False,
    schedule=None,
    default_args=default_args,
    params={
        "experiment": Param("airflow-pipeline"),
        "bucket": Param("input-data"),
        "file": Param("customer-segmentation.csv"),
        "output_path": Param("airflow-data/processed/"),
        "test_size": Param("0.2"),
        "random_state": Param("42"),
        "lr_max_iter": Param("100"),
        "lr_tol": Param("0.0001"),
        "rf_n_estimators": Param("100"),
        "rf_max_depth": Param("10"),
        "gbm_n_estimators": Param("100"),
        "gbm_learning_rate": Param("0.1"),
        "min_f1_macro": Param("0.5"),
        "min_precision_macro": Param("0.5"),
        "f1_drift_factor": Param("0.5"),
        "model_name": Param("cusomer-segmentation"),
    }
) as dag:

    eda = KubernetesPodOperator(
        task_id="eda",
        image=IMAGE,
        cmds=["python", "eda.py"],
        arguments=[
            "--experiment-name-mlflow", "{{ params.experiment }}",
            "--bucket-name", "{{ params.bucket }}",
            "--filename", "{{ params.file }}",
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
            "--output-path", "{{ params.output_path }}",
            "--test-size", "{{ params.test_size }}",
            "--random-state", "{{ params.random_state }}",
        ],
        namespace="airflow",
        secrets=SECRETS,
        env_vars=COMMON_ENV,
        do_xcom_push=True,
        get_logs=True,
        on_finish_action="delete_succeeded_pod",
    )

    with TaskGroup("train_models") as train_models:

        lr = KubernetesPodOperator(
            task_id="train_lr",
            image=IMAGE,
            cmds=["python", "train.py"],
            arguments=[
                "--experiment-name-mlflow", "{{ params.experiment }}",
                "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess')['run_id'] }}",
                "--model-type", "lr",
                "--random-state", "{{ params.random_state }}",
                "--lr-max-iter", "{{ params.lr_max_iter }}",
                "--lr-tol", "{{ params.lr_tol}}",
            ],
            namespace="airflow",
            secrets=SECRETS,
            env_vars=COMMON_ENV,
            do_xcom_push=True,
            get_logs=True,
            on_finish_action="delete_succeeded_pod",
        )

        rf = KubernetesPodOperator(
            task_id="train_rf",
            image=IMAGE,
            cmds=["python", "train.py"],
            arguments=[
                "--experiment-name-mlflow", "{{ params.experiment }}",
                "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess')['run_id'] }}",
                "--model-type", "rf",
                "--random-state", "{{ params.random_state }}",
                "--rf-n-estimators", "{{ params.rf_n_estimators }}",
                "--rf-max-depth", "{{ params.rf_max_depth }}",
            ],
            namespace="airflow",
            secrets=SECRETS,
            env_vars=COMMON_ENV,
            do_xcom_push=True,
            get_logs=True,
            on_finish_action="delete_succeeded_pod",
        )

        gbm = KubernetesPodOperator(
            task_id="train_gbm",
            image=IMAGE,
            cmds=["python", "train.py"],
            arguments=[
                "--experiment-name-mlflow", "{{ params.experiment }}",
                "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess')['run_id'] }}",
                "--model-type", "gbm",
                "--random-state", "{{ params.random_state }}",
                "--gbm-n-estimators", "{{ params.gbm_n_estimators }}",
                "--gbm-learning-rate", "{{ params.gbm_learning_rate }}",
            ],
            namespace="airflow",
            secrets=SECRETS,
            env_vars=COMMON_ENV,
            do_xcom_push=True,
            get_logs=True,
            on_finish_action="delete_succeeded_pod",
        )

    evaluate = KubernetesPodOperator(
        task_id="evaluate",
        image=IMAGE,
        cmds=["python", "evaluation.py"],
        arguments=[
            "--experiment-name-mlflow", "{{ params.experiment }}",
            "--training-results", "{{ ti.xcom_pull(task_ids=['train_models.train_lr','train_models.train_rf','train_models.train_gbm']) | tojson }}",
            "--preprocessing-run-id", "{{ ti.xcom_pull(task_ids='preprocess')['run_id'] }}",
            "--min-f1-macro", "{{ params.min_f1_macro }}",
            "--min-precision-macro", "{{ params.min_precision_macro }}",
            "--f1-drift-factor", "{{ params.f1_drift_factor }}",
        ],
        namespace="airflow",
        secrets=SECRETS,
        env_vars=COMMON_ENV,
        do_xcom_push=True,
        get_logs=True,
        on_finish_action="delete_succeeded_pod",
    )

    gate = ShortCircuitOperator(
        task_id="promotion_gate",
        python_callable=promotion_gate,
    )

    promote = KubernetesPodOperator(
        task_id="promote_model",
        image=IMAGE,
        cmds=["python", "promote.py"],
        arguments=[
            "--model-name", "{{ params.model_name }}",
            "--best-run-id", "{{ ti.xcom_pull(task_ids='evaluate') }}",
        ],
        namespace="airflow",
        secrets=SECRETS,
        env_vars=COMMON_ENV,
        get_logs=True,
        on_finish_action="delete_succeeded_pod",
    )


    eda >> preprocess >> train_models >> evaluate >> gate >> promote

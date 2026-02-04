import os
import mlflow
from mlflow.tracking import MlflowClient
import json

client = MlflowClient()

model_name = "customer_segmentation"

with open("/airflow/xcom/return.json") as f:
    data = json.load(f)

run_id = data["best_run_id"]

model_uri = f"runs:/{run_id}/model"

# Register
result = mlflow.register_model(model_uri, model_name)

# Wait until ready
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Production",
    archive_existing_versions=True,
)

print(f"Model {model_name} promoted to Production")

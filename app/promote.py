import os
import mlflow
from mlflow.tracking import MlflowClient
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--best-run-id", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    client = MlflowClient()
    model_name = args.model_name
    run_id = args.best_run_id

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



if __name__ == "__main__":
    main()
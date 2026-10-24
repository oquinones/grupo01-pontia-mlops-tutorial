import os

import mlflow

MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000/")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "grupo01-adult-income")

mlflow.set_tracking_uri(MLFLOW_URL)

local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{EXPERIMENT_NAME}/latest", dst_path="./models"
)

print(f"Modelo descargado en: {local_path}")

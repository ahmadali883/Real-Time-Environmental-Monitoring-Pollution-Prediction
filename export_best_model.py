import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import joblib
import os

# Set the MLflow tracking URI to the running MLflow server
mlflow.set_tracking_uri("http://localhost:5000")  # Update if your server is running elsewhere

# Parameters
EXPERIMENT_NAME = "AQI_Prediction_LSTM_Tuner"  # Ensure this matches your experiment name
BEST_RUN_LIMIT = 1  # Number of top runs to consider

# Initialize MLflow client
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if not experiment:
    print(f"Experiment '{EXPERIMENT_NAME}' not found.")
    exit()

# Search for the best run based on evaluation RMSE
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],  # Adjust metric name if different
    max_results=BEST_RUN_LIMIT
)

if not runs:
    print("No runs found in the experiment.")
    exit()

best_run = runs[0]
best_run_id = best_run.info.run_id
print(f"Best Run ID: {best_run_id}")

# Define the model URI
model_uri = f"runs:/{best_run_id}/model"

# Load the best model
try:
    best_model = mlflow.keras.load_model(model_uri)
    print("Best LSTM model loaded successfully.")
except Exception as e:
    print(f"Failed to load the model: {e}")
    exit()

# Save the model locally in Keras format (recommended over HDF5)
model_save_path = "best_lstm_model.keras"
best_model.save(model_save_path)
print(f"Best LSTM model saved to '{model_save_path}'.")

# Define the scaler artifact path
scaler_artifact_path = "scaler.joblib"  # Ensure this matches how you logged it during training

# Download the scaler artifact
try:
    scaler_local_path = mlflow.artifacts.download_artifacts(
        run_id=best_run_id,
        artifact_path=scaler_artifact_path
    )
    # Load the scaler using joblib
    scaler = joblib.load(scaler_local_path)
    # Save the scaler locally
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved to 'scaler.joblib'.")
except Exception as e:
    print(f"Failed to load the scaler: {e}")
    exit()

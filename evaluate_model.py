import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# Set the MLflow tracking URI to the running MLflow server
mlflow.set_tracking_uri("http://localhost:5000")  # Update if different

# Parameters
EXPERIMENT_NAME = "AQI_Prediction_LSTM_Tuner"  # Ensure this matches your experiment name
BEST_RUN_LIMIT = 1  # Number of top runs to consider

# Load preprocessed data
df = pd.read_csv('processed_data.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Choose a location for evaluation (e.g., Los Angeles)
location = 'Los Angeles'
location_df = df[df['location'] == location]

# Define the target variable
ts = location_df['aqi'].values.reshape(-1, 1)

# Check if there's enough data
if len(ts) < 10:  # Adjust threshold as needed
    print("Not enough data points for evaluation.")
    exit()

# Load the scaler
try:
    scaler = joblib.load('scaler.joblib')
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Failed to load scaler: {e}")
    exit()

# Scale the data
ts_scaled = scaler.transform(ts)

# Prepare the dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1  # Ensure this matches the training
X, Y = create_dataset(ts_scaled, look_back)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Initialize MLflow client
client = mlflow.tracking.MlflowClient()
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

# Load the best model
model_uri = f"runs:/{best_run_id}/model"
try:
    best_model = mlflow.keras.load_model(model_uri)
    print("Best LSTM model loaded successfully.")
except Exception as e:
    print(f"Failed to load the model: {e}")
    exit()

# Make predictions
predictions = best_model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
Y_test_actual = scaler.inverse_transform([Y_test])

# Evaluate the model
rmse = np.sqrt(mean_squared_error(Y_test_actual[0], predictions[:,0]))
mae = mean_absolute_error(Y_test_actual[0], predictions[:,0])

print(f'LSTM Model Evaluation Metrics:')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# Log evaluation metrics to MLflow
with mlflow.start_run(run_id=best_run_id, nested=True):
    mlflow.log_metric("evaluation_rmse", rmse)
    mlflow.log_metric("evaluation_mae", mae)

# Plot Actual vs Predicted AQI
plt.figure(figsize=(12,6))
plt.plot(Y_test_actual[0], label='Actual AQI')
plt.plot(predictions[:,0], label='Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.xlabel('Time Steps')
plt.ylabel('AQI')
plt.legend()
plt.savefig('actual_vs_predicted_aqi.png')
plt.show()

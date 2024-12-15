import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
import joblib  # Import joblib for saving the scaler

# Define HyperModel for LSTM
class AQILSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            input_shape=(look_back, 1),
            return_sequences=False
        ))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(1))
        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        return model

# Load preprocessed data
df = pd.read_csv('processed_data.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Choose a location for modeling (e.g., Los Angeles)
location = 'Los Angeles'
location_df = df[df['location'] == location]

# Define the target variable
ts = location_df['aqi'].values.reshape(-1, 1)

# Check if there's enough data
if len(ts) < 10:  # Adjust threshold as needed
    print("Not enough data points for LSTM modeling.")
    exit()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
ts_scaled = scaler.fit_transform(ts)

# Prepare the dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1  # Adjust based on data availability
X, Y = create_dataset(ts_scaled, look_back)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("AQI_Prediction_LSTM_Tuner")

# Define the tuner
hypermodel = AQILSTMHyperModel()
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='AQI_LSTM_Tuning'
)

# Start tuning
tuner.search(X_train, Y_train, epochs=20, batch_size=1, validation_data=(X_test, Y_test))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete.
Optimal number of units: {best_hps.get('units')}
Optimal dropout rate: {best_hps.get('dropout')}
""")

# Train the best model
model = tuner.hypermodel.build(best_hps)
model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=0, validation_data=(X_test, Y_test))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
Y_test_actual = scaler.inverse_transform([Y_test])

# Evaluate the model
rmse = np.sqrt(mean_squared_error(Y_test_actual[0], predictions[:,0]))

# Log the model and scaler with MLflow
with mlflow.start_run():
    mlflow.log_param("units", best_hps.get('units'))
    mlflow.log_param("dropout", best_hps.get('dropout'))
    mlflow.log_metric("rmse", rmse)
    mlflow.keras.log_model(model, "model")
    
    # Save the scaler locally
    scaler_filename = 'scaler.joblib'
    joblib.dump(scaler, scaler_filename)
    
    # Log the scaler as an artifact
    mlflow.log_artifact(scaler_filename)
    
    print(f'LSTM Model RMSE: {rmse}')
    print(f'Scaler saved and logged to MLflow as {scaler_filename}')

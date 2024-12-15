import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

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

look_back = 1  # Use past 'look_back' hours to predict the next hour
X, Y = create_dataset(ts_scaled, look_back)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Update if different
mlflow.set_experiment("AQI_Prediction_LSTM")

with mlflow.start_run():
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model
    model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=0)
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    Y_test_actual = scaler.inverse_transform([Y_test])
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(Y_test_actual[0], predictions[:,0]))
    mlflow.log_param("look_back", look_back)
    mlflow.log_metric("rmse", rmse)
    
    # Log the model
    mlflow.keras.log_model(model, "model")
    
    print(f'LSTM Model RMSE: {rmse}')

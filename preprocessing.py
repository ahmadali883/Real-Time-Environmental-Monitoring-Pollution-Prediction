import os
import json
import pandas as pd

DATA_DIR = 'data'

def load_data(data_dir):
    records = []
    json_files = [file for file in os.listdir(data_dir) if file.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in '{data_dir}' directory.")
    
    for file in json_files:
        file_path = os.path.join(data_dir, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                records.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file}: {e}")
        except Exception as e:
            print(f"Unexpected error reading file {file}: {e}")
    
    print(f"Loaded {len(records)} records from JSON files.")
    return pd.DataFrame(records)

def preprocess_data(df):
    print("Starting preprocessing...")
    
    if df.empty:
        print("DataFrame is empty. Exiting preprocessing.")
        return df
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with invalid timestamps
    initial_count = len(df)
    df = df.dropna(subset=['timestamp'])
    dropped_timestamps = initial_count - len(df)
    if dropped_timestamps > 0:
        print(f"Dropped {dropped_timestamps} rows due to invalid timestamps.")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Handle missing AQI values
    initial_count = len(df)
    df = df.dropna(subset=['aqi'])  # Ensure AQI is present
    dropped_aqi = initial_count - len(df)
    if dropped_aqi > 0:
        print(f"Dropped {dropped_aqi} rows due to missing AQI.")
    
    # Forward fill other missing values
    df = df.ffill()
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Lag features (e.g., previous hour AQI)
    df['aqi_lag_1'] = df.groupby('location')['aqi'].shift(1)
    
    # Drop rows with NaN in 'aqi_lag_1'
    initial_count = len(df)
    df = df.dropna(subset=['aqi_lag_1'])
    dropped_lag = initial_count - len(df)
    if dropped_lag > 0:
        print(f"Dropped {dropped_lag} rows due to NaN in 'aqi_lag_1'.")
    
    print(f"Preprocessing completed. Final dataset has {len(df)} records.")
    return df

if __name__ == '__main__':
    df = load_data(DATA_DIR)
    df = preprocess_data(df)
    
    if not df.empty:
        df.to_csv('processed_data.csv', index=False)
        print('Data preprocessing completed and saved to processed_data.csv')
    else:
        print('No data to save after preprocessing.')

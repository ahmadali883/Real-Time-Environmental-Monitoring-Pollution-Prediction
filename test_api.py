import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='test_api.log',
                    format='%(asctime)s %(levelname)s:%(message)s')

# Define the API endpoint
url = 'http://localhost:5000/predict'

# Define the polling interval (e.g., every hour)
polling_interval = 5  # in seconds

def fetch_latest_history():
    # Implement logic to fetch the latest 'look_back' AQI values from your data source
    # For demonstration, we'll use dummy data
    # Replace this with actual data fetching logic
    return [5]  # Replace with the actual AQI history

def make_prediction(history):
    payload = {
        "history": history
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            predicted_aqi = response.json().get('predicted_aqi')
            logging.info(f"Predicted AQI: {predicted_aqi}")
            print(f"Predicted AQI: {predicted_aqi}")
        else:
            error = response.json().get('error', 'Unknown error')
            logging.error(f"Error from API: {error}")
            print(f"Error: {error}")
    except Exception as e:
        logging.error(f"Exception during API call: {e}")
        print(f"Exception: {e}")

if __name__ == "__main__":
    while True:
        history = fetch_latest_history()
        if history:
            make_prediction(history)
        else:
            logging.warning("No history data available for prediction.")
            print("No history data available for prediction.")
        time.sleep(polling_interval)

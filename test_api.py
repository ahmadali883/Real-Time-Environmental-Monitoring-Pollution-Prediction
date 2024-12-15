import requests

# Define the API endpoint
url = 'http://localhost:5000/predict'

# Define the payload
payload = {
    "history": [5]
}

# Make the POST request
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    print("Predicted AQI:", response.json()['predicted_aqi'])
else:
    print("Error:", response.json())

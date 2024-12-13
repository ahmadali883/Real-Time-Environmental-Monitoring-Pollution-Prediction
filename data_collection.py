import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
AIRVISUAL_API_KEY = os.getenv('AIRVISUAL_API_KEY')

# Define locations (latitude and longitude)
LOCATIONS = [
    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
    {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
    # Add more locations as needed
]

# Create data directory if it doesn't exist
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_openweathermap_data(location):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={location["lat"]}&lon={location["lon"]}&appid={OPENWEATHERMAP_API_KEY}&units=metric'
    response = requests.get(url)
    return response.json()

def fetch_airvisual_data(location):
    url = f'http://api.airvisual.com/v2/nearest_city?lat={location["lat"]}&lon={location["lon"]}&key={AIRVISUAL_API_KEY}'
    response = requests.get(url)
    return response.json()

def save_data():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for loc in LOCATIONS:
        ow_data = fetch_openweathermap_data(loc)
        av_data = fetch_airvisual_data(loc)
        
        # Extract relevant weather data
        weather_record = {
            'timestamp': datetime.now().isoformat(),
            'location': loc['name'],
            'temperature': ow_data.get('main', {}).get('temp'),
            'humidity': ow_data.get('main', {}).get('humidity'),
            'pressure': ow_data.get('main', {}).get('pressure'),
            'wind_speed': ow_data.get('wind', {}).get('speed'),
            'wind_deg': ow_data.get('wind', {}).get('deg'),
            'clouds': ow_data.get('clouds', {}).get('all'),
            'visibility': ow_data.get('visibility'),
            'weather_description': ow_data.get('weather', [{}])[0].get('description'),
            'uv_index': ow_data.get('uvi'),  # Ensure 'uvi' is included in API response
            # Add more fields as needed
        }
        
        # Save Weather Data
        ow_file = os.path.join(DATA_DIR, f'ow_{loc["name"]}_{timestamp}.json')
        with open(ow_file, 'w') as f:
            json.dump(weather_record, f)
        
        # Save AirVisual Data (if needed)
        av_file = os.path.join(DATA_DIR, f'av_{loc["name"]}_{timestamp}.json')
        with open(av_file, 'w') as f:
            json.dump(av_data, f)

if __name__ == '__main__':
    save_data()
    print(f'Data fetched and saved at {datetime.now()}')

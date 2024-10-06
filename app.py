import streamlit as st
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import import_ipynb  # Import the Jupyter notebook as a module
from Hacakthon import *  # Import everything from your Jupyter notebook
import os

# Define your dataset and model classes
class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.features = data[['lat', 'lon', 'year', 'doy', 'uv_index', 'temp_2m',
                              'dew_2m', 'humidity_2m', 'surface_pressure',
                              'wind_speed_2m', 'surface_soil_wetness',
                              'root_soil_wetness']].values
        self.targets = data[['precipitation_mm', 'temp_2m', 'dew_2m',
                             'humidity_2m', 'surface_pressure',
                             'wind_speed_2m', 'surface_soil_wetness',
                             'root_soil_wetness']].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class AFNO1DNet(nn.Module):
    def __init__(self, in_chans=12, out_chans=8, hidden_size=64):
        super(AFNO1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_chans, out_channels=hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=out_chans, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(out_chans)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        return x

# Thresholds and alerts
THRESHOLDS = {
    'uv_index': 8,
    'temp_2m': 35,
    'dew_2m': 2,
    'precipitation_mm': 50,
    'humidity_2m': 90,
    'surface_pressure': 1000,
    'wind_speed_2m': 20,
    'surface_soil_wetness': 0.9,
    'root_soil_wetness': 0.9
}

ALERT_MESSAGES = {
    'uv_index': "High UV Index Alert! Please take necessary precautions.",
    'temp_2m': "Heatwave Alert! Temperature is predicted to rise above 35°C.",
    'dew_2m': "Frost Alert! Dew point is very low.",
    'precipitation_mm': "Heavy Rainfall Alert! Precipitation is predicted to exceed 50mm.",
    'humidity_2m': "High Humidity Alert! Humidity levels are above 90%.",
    'surface_pressure': "Low Pressure Alert! There may be a storm or extreme weather.",
    'wind_speed_2m': "Strong Wind Alert! Wind speeds exceed 20 m/s.",
    'surface_soil_wetness': "Flood Risk Alert! Surface soil wetness is very high.",
    'root_soil_wetness': "Waterlogging Alert! Root soil wetness is very high."
}

# Prediction and alert generation functions
def generate_alerts(predictions):
    alerts = []
    for i, (var, threshold) in enumerate(THRESHOLDS.items()):
        if i < len(predictions):  # Ensure no out-of-bounds indexing
            predicted_value = predictions[i]
            if predicted_value > threshold:
                alerts.append(ALERT_MESSAGES[var])
    return alerts

def predict(model, lat, lon, year, doy):
    model.eval()
    # Construct input tensor
    input_tensor = torch.tensor([[lat, lon, year, doy, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(1)  # Add the channel dimension
    
    with torch.no_grad():
        output = model(input_tensor.permute(0, 2, 1))  # Rearranging for Conv1d
    
    return output.squeeze().numpy()

# Streamlit UI
st.title('AFNO Weather Alert System')

# Load the model (add your saved model path if necessary)
model = AFNO1DNet(in_chans=12, out_chans=8)  # Define model structure
# model.load_state_dict(torch.load('afno1dnet_model.pth'))  # Uncomment if you have a saved model

# Load dataset
dataset_path = 'df_combined.csv'  # Update this path if necessary
if not os.path.exists(dataset_path):
    st.error(f"Dataset not found at {dataset_path}")
else:
    dataset = WeatherDataset(dataset_path)

# Take user input
lat = st.number_input('Enter Latitude:', format="%.6f")
lon = st.number_input('Enter Longitude:', format="%.6f")
year = st.number_input('Enter Year:', min_value=2000, max_value=2100, step=1)
doy = st.number_input('Enter Day of Year (1-365):', min_value=1, max_value=365, step=1)

# Prediction and alert generation on user input
if st.button('Predict Weather and Generate Alerts'):
    if lat and lon and year and doy:
        predictions = predict(model, lat, lon, year, doy)

        # Display predictions
        st.write("Predicted Weather Conditions:")
        for i, (var, value) in enumerate(zip(THRESHOLDS

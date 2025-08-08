"""
Script: generate_background_data.py

This script generates and exports background (non-earthquake) waveform data from InfluxDB
for use in machine learning models (e.g., CNNs) that classify earthquake events.

Key steps:
1. Load earthquake timestamps from a CSV file.
2. Define a full hourly time range from the earliest to latest earthquake.
3. Exclude all hours within ±1 hour of any earthquake (configurable).
4. Randomly sample 1000 valid background hours from the remaining time range.
5. For each sampled hour, query InfluxDB to fetch waveform data (±15s to ±45s window).
6. Store the retrieved data (if available) in a dictionary with timestamps.
7. Save the data to a `.pkl` file for later use.

Modules required:
- pandas, numpy, pickle, pathlib, datetime
- paros_data_grabber.query_influx_data (custom module for InfluxDB queries)
- tqdm (for progress bar display)

Note: You must replace the InfluxDB password placeholder with the actual password.

Ethan Gelfand, 08/06/2025
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import pickle
import os
from pathlib import Path
from paros_data_grabber import query_influx_data
from tqdm import tqdm

def generate_background_hours(start_time, end_time, earthquake_datetimes, buffer_hours=1):
    """Generate hourly timestamps excluding ±buffer_hours around earthquake times."""
    all_hours = pd.date_range(start=start_time, end=end_time, freq='h')
    excluded = pd.DatetimeIndex([])

    for dt in earthquake_datetimes:
        buffer_range = pd.date_range(dt - pd.Timedelta(hours=buffer_hours),
                                     dt + pd.Timedelta(hours=buffer_hours),
                                     freq='h')
        excluded = excluded.union(buffer_range)

    return all_hours.difference(excluded).unique()

def sample_background_hours(available_hours, num_samples, seed=42):
    """Randomly sample background hours."""
    num_samples = min(num_samples, len(available_hours))
    rng = np.random.default_rng(seed)
    return pd.DatetimeIndex(rng.choice(available_hours, size=num_samples, replace=False))

# --- Load earthquake data ---
earthquake_data = pd.read_csv("EarthQuakeData.csv")
earthquake_datetimes = pd.to_datetime(earthquake_data['time'])


# --- Define range of interest ---
start_time = earthquake_datetimes.min().floor('h')
end_time = earthquake_datetimes.max().ceil('h')

# --- Get valid background hours ---
background_hours = generate_background_hours(start_time, end_time, earthquake_datetimes, buffer_hours=1)

# --- Sample N background hours ---
selected_hours = sample_background_hours(background_hours, num_samples=1000)

print(f"Selected {len(selected_hours)} background hours.")

# --- InfluxDB fetch config ---
box_id = "parost2"
sensor_id = "141929"
time_before = timedelta(seconds=15)
time_after = timedelta(seconds=45)

# --- Fetch and store ---
all_data = {}
event_counter = 1

for idx in tqdm(range(len(selected_hours)), desc="Processing Events", colour="green"):
    timestamp = selected_hours.sort_values()[idx]
    try:
        start_t = (timestamp - time_before).strftime("%Y-%m-%dT%H:%M:%S")
        end_t = (timestamp + time_after).strftime("%Y-%m-%dT%H:%M:%S")

        data = query_influx_data(
            start_time=start_t,
            end_time=end_t,
            box_id=box_id,
            sensor_id=sensor_id,
            password="*****",  # Replace with actual password
        )

        if not data:
            tqdm.write(f"No data returned for hour {idx+1} at {timestamp}")
            continue

        data_arrays = {key: df_.values for key, df_ in data.items()}

        all_data[f"background_{event_counter:04d}"] = {
            'waveform': data_arrays,
            'timestamp': timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        }

        event_counter += 1

    except Exception as e:
        tqdm.write(f"Failed on hour {idx+1}: {e}")
        continue

# --- Save as .mat file ---
dir = "Exported_Paros_Data"
os.makedirs(dir, exist_ok=True)
output_file = "Exported_Paros_Data/background_data.pkl"

with open(output_file, 'wb') as f:
            pickle.dump(all_data, f)
        
print(f"[Done] Data saved to {"Exported_Paros_Data/background_data.pkl"}")

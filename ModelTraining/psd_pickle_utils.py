"""
PSD Extraction Utilities for Earthquake/Infrasound Event Data

This script contains utility functions for loading and extracting Power Spectral
Density (PSD) data from pre-processed `.pkl` files. The PSD data is organized in
a nested structure of events and time windows.

Functions:
- extract_psd_array(psd_struct, num_windows=11):
    Extracts PSD power arrays from a structured dictionary of events.
    Each event is expected to contain a fixed number of time windows
    (e.g., 11), each with a 'power' field.
    Returns a NumPy array of shape (events, windows, freq_bins).

- load_pickle_data(path):
    Loads a `.pkl` file and extracts the 'psdResults' field, if present.
    This mimics loading MATLAB `.mat` files with nested PSD results.

Intended Use:
- For extracting model-ready PSD features from structured pickle files
  exported from MATLAB or other pre-processing pipelines.

Dependencies:
- NumPy
- Pickle

Ethan Gelfand, 08/06/2025
"""

import numpy as np
import pickle

def extract_psd_array(psd_struct, num_windows=11):
    data = []

    # Filter keys that correspond to events
    event_keys = [k for k in psd_struct.keys() if k.startswith("event_")]

    for key in event_keys:
        event = psd_struct[key]

        # Filter and sort window keys
        window_keys = [k for k in event.keys() if k.startswith("window_")]
        window_keys = sorted(window_keys, key=lambda x: int(x.split('_')[1]))[:num_windows]

        event_data = []
        for wk in window_keys:
            try:
                power = np.array(event[wk]["power"])
                if power.ndim == 2:
                    power = power[0, :]  # Take first channel if 2D
                event_data.append(power)
            except (KeyError, TypeError):
                continue

        if len(event_data) == num_windows:
            data.append(np.stack(event_data))  # (windows, freq_bins)

    return np.stack(data) if data else np.array([])  # (events, windows, freq_bins)

def load_pickle_data(path):
    """
    Loads a pickle file and returns the 'psdResults' field if present,
    similar to how loadmat(...)[‘psdResults’] works.

    Parameters:
        path (str): Path to the .pkl file

    Returns:
        dict: Parsed PSD data (e.g., events with windows and power values)
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # extract 'psdResults' field if it exists
    if isinstance(data, dict) and 'psdResults' in data:
        return data['psdResults']
    return data

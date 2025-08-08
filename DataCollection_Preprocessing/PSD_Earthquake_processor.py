"""
Script: process_earthquake_waveforms.py

This script processes earthquake waveform data (stored in a pickle file) to compute
windowed Power Spectral Density (PSD) features for each valid event. The output includes
metadata and PSD values per window, and is saved to a new `.pkl` file for later use in
machine learning models.

Main Workflow:
1. Load earthquake event data from `EarthQuakeEvents.pkl`.
2. For each event:
   - Extract waveform data from the `parost2_141929` sensor.
   - Resample the waveform from 20 Hz to 100 Hz.
   - Apply DC-blocking and high-pass filtering (preprocessing).
   - Zero-pad slightly short signals if needed.
   - Segment into overlapping 10-second windows (50% overlap).
   - Compute Welch PSD for each window (frequencies ≤ 10 Hz).
   - Store PSD values and event metadata.
3. Skip events with too few samples or windows (<11).
4. Save the final dictionary of PSDs and metadata to `PSD_Windows_Earthquake_100Hz.pkl`.

Parameters:
- `fs_in = 20`: Input sample rate (Hz)
- `fs_out = 100`: Output resample rate (Hz)
- `delta_t = 10`: Window duration (seconds)
- `overlap = 0.5`: 50% window overlap

Dependencies:
- numpy, tqdm, pickle, os, Preprocessing_fun script
- Assumes waveform data is structured with `waveform['parost2_141929']` and includes `metadata`

Note: This script is nearly identical to the background processor, but includes event metadata
in the output and handles earthquake-labeled waveform data.

Ethan Gelfand, 08/06/2025
"""

import pickle
import numpy as np
import os
from tqdm import tqdm
from Preprocessing_fun import preprocess, welch_psd, safe_resample

## --- load pickle file --- ##
file_path = "Exported_Paros_Data/EarthQuakeEvents.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

## --- start processing earthquake data --- ##

# parameters
fs_in = 20          # original sampling frequency (Hz)
fs_out = 100        # target sampling frequency (Hz)
delta_t = 10        # window length in seconds
overlap = 0.5       # 50% overlap
fs = fs_out
psdResults = {}
goodEventCounter = 1

eventNames = list(data.keys())
for eventName in tqdm(eventNames, colour="green"):
    try:
        eventStuct = data[eventName]
        waveform = eventStuct['waveform']['parost2_141929'][:, -1]
        waveform = np.array(waveform, dtype=float)
        metadata = eventStuct['metadata']

        waveform = safe_resample(waveform, fs_in, fs_out)

        window_size = int(fs * delta_t) 
        stride = int(window_size * (1 - overlap))

        if len(waveform) < window_size:
            tqdm.write(f'Skipping {eventName}: not enough samples ({len(waveform)})')
            continue

        waveform = preprocess(waveform, fs)

        # zero pad if the wave form is within 95% of the expected length of 6000
        if 5700 <= len(waveform) < 6000:
                waveform = np.pad(waveform, (0, 6000 - len(waveform)), mode="constant")

        num_windows = (len(waveform) - window_size) // stride + 1

        if num_windows < 11:
            tqdm.write(f'Skipping {eventName}: only {num_windows} windows (need at least 11)')
            continue

        eventPSD = {'metadata': metadata}
        for w in range(num_windows):
            idx_start = w * stride
            idx_end = idx_start + window_size
            segment = waveform[idx_start:idx_end]

            pxx, f = welch_psd(segment, fs)
            winName = f'window_{w+1:03d}'

            eventPSD[winName] = {'power': pxx, 'frequency': f}
                
        keyName = f'event_{goodEventCounter:03d}'
        psdResults[keyName] = eventPSD
        goodEventCounter += 1
            
    except Exception as e:
        tqdm.write(f'Error processing {eventName}: {e}')
        
# Save as pickle
output_path = os.path.join(os.getcwd(), "Exported_Paros_Data/PSD_Windows_Earthquake_100Hz.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(psdResults, f)

print(f'Saved PSDs with metadata to {output_path}')
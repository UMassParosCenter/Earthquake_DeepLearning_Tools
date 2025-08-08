"""
Script: process_background_waveforms.py

This script processes raw background waveform data (from a previously saved pickle file)
and computes windowed Power Spectral Density (PSD) features for each valid background event.
The output is saved as a new `.pkl` file containing PSD windows suitable for machine learning tasks.

Main steps:
1. Load background waveform data from `background_data.pkl`.
2. For each event:
   - Extract the waveform from the specified sensor.
   - Resample from 20 Hz to 100 Hz.
   - DC-block and high-pass filter (preprocessing).
   - Zero-pad if slightly short.
   - Segment into overlapping windows (10 seconds, 50% overlap).
   - Compute Welch PSD for each window (up to 10 Hz).
   - Store results with associated window names.
3. Save processed PSD data into `PSD_Windows_Background_100Hz.pkl`.

Key Parameters:
- `fs_in = 20`: Original sample rate (Hz)
- `fs_out = 100`: Target resample rate (Hz)
- `delta_t = 10`: Window length in seconds
- `overlap = 0.5`: 50% overlap between windows

Requirements:
- numpy, tqdm, pickle, os, Preprocessing_fun script
- Data must include `parost2_141929` waveform key

Note: Events with fewer than 11 windows or insufficient samples are skipped.

Ethan Gelfand, 08/06/2025
"""

import pickle
import numpy as np
import os
from tqdm import tqdm
from Preprocessing_fun import preprocess, welch_psd, safe_resample


## --- Load pickle file --- ##
file_path = "Exported_Paros_Data/background_data.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

## --- Start processing background data --- ##
fs_in = 20          # Original sampling frequency (Hz)
fs_out = 100        # Target sampling frequency (Hz)
delta_t = 10        # Window length in seconds
overlap = 0.5       # 50% overlap
fs = fs_out
psdResults = {}
goodEventCounter = 1

mean_psds = [] # array to store means for each event
eventNames = list(data.keys())
for eventName in tqdm(eventNames, colour="green"):
    try:
        eventStuct = data[eventName]
        waveform = eventStuct['waveform']['parost2_141929'][:, -1]
        waveform = np.array(waveform, dtype=float)

        # Resample the waveform to the target frequency
        waveform = safe_resample(waveform, fs_in, fs_out)

        # Split the waveform into windows
        window_size = int(fs * delta_t) 
        stride = int(window_size * (1 - overlap))

        if len(waveform) < window_size:
            tqdm.write(f'Skipping {eventName}: not enough samples ({len(waveform)})')
            continue

        # Preprocess
        waveform = preprocess(waveform, fs)

        # zero pad if the wave form is within 95% of the expected length of 6000
        if 5700 <= len(waveform) < 6000:
                waveform = np.pad(waveform, (0, 6000 - len(waveform)), mode="constant")

        num_windows = (len(waveform) - window_size) // stride + 1

        if num_windows < 11:
            tqdm.write(f'Skipping {eventName}: only {num_windows} windows (need at least 11)')
            continue

        eventPSD = {}
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
output_path = os.path.join(os.getcwd(), "Exported_Paros_Data/PSD_Windows_Background_100Hz.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(psdResults, f)

print(f'Saved PSDs to {output_path}')

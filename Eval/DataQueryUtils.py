"""
Paros Infrasound PSD Feature Extraction for Machine Learning
============================================================

This script provides functionality for querying and preprocessing waveform data 
from a Paros infrasound sensor for use in machine learning models. It includes 
two main functions:

1. live_stream_query_for_model():
   - Queries the most recent 60-second waveform segment from a Paros sensor via InfluxDB.
   - Resamples and preprocesses the waveform.
   - Splits each 60-second segment into eleven 10-second windows with 50% overlap.
   - Computes Welch PSD for each window and stacks them into a flattened feature vector.
   - Normalizes the flattened PSD vector using the provided mean and standard deviation 
     calculated from training data.

2. psd_vectors_from_range():
   - Iterates through a user-defined date/time range in 60-second increments.
   - For each segment, performs the same resampling, preprocessing, PSD computation, 
     windowing, flattening, and normalization as above.
   - Returns a list of timestamped normalized feature vectors for batch inference or analysis.

These functions rely on custom utilities:
- query_influx_data: for data retrieval from InfluxDB
- preprocess: for waveform preprocessing
- safe_resample: for robust resampling
- welch_psd: for power spectral density calculation

The output vectors are designed as inputs for trained machine learning models, such as
earthquake event classifiers.

Ethan Gelfand, 08/06/2025
"""


import numpy as np
from datetime import datetime, timedelta, timezone
from paros_data_grabber import query_influx_data
from Preprocessing_fun import preprocess, welch_psd, safe_resample

def live_stream_query_for_model(
    sensor_id="141929",
    box_id="parost2",
    password="*****", # Replace with actual password
    fs_in=20,
    fs_out=100,
    total_duration=60,
    window_duration=10,
    overlap=0.5,
    mean=None,
    std=None
):
    try:
        # Get timestamps for 60-second segment
        end_time = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
        start_time = end_time - timedelta(seconds=total_duration)

        start_str = start_time.isoformat(timespec='seconds')
        end_str = end_time.isoformat(timespec='seconds')
        print(f"Querying data from {start_str} to {end_str}")

        # Query data
        data = query_influx_data(
            start_time=start_str,
            end_time=end_str,
            box_id=box_id,
            sensor_id=sensor_id,
            password=password
        )

        key = f"{box_id}_{sensor_id}"
        waveform = data.get(key)
        if waveform is None or waveform.empty:
            print("No data received")
            return None

        samples = waveform['value'].values
        x = safe_resample(samples, fs_in, fs_out)
        x = preprocess(x, fs_out)

        # Pad only if close to 6000 ~95% or greater
        if 5700 <= len(x) < 6000:
            pad_len = 6000 - len(x)
            x = np.pad(x, (0, pad_len), mode='constant')
            print(f"Padded waveform with {pad_len} zeros to reach 6000 samples.")

        # Early exit if not enough data
        if len(x) < 6000:
            print(f"Insufficient data after resampling: {len(x)} samples")
            return None

        # PSD windowing
        win_length = int(window_duration * fs_out)
        step = int(win_length * (1 - overlap))
        n_windows = (len(x) - win_length) // step + 1

        if n_windows != 11:
            print(f"Number of windows found: {n_windows} (expected 11). Skipping.")
            return None

        psd_list = []
        for i in range(n_windows):
            start_idx = i * step
            end_idx = start_idx + win_length
            window_data = x[start_idx:end_idx]
            pxx, freqs = welch_psd(window_data, fs_out)
            psd_list.append(pxx)

        psd_array = np.vstack(psd_list)
        psd_vector = psd_array.flatten()

        log_pxx = np.log10(psd_vector + 1e-10)
        if mean is not None and std is not None:
            z_pxx = (log_pxx - mean) / (std + 1e-6)
        else:
            z_pxx = log_pxx

        return z_pxx.astype(np.float32)

    except Exception as e:
        print("Error during stream:", e)
        return None

        
def psd_vectors_from_range(
    start_time,
    end_time,
    sensor_id="141929",
    box_id="parost2",
    password="*****", # Replace with actual password
    fs_in=20,
    fs_out=100,
    window_duration=10,
    overlap=0.5,
    mean=None,
    std=None
):
    results = []

    duration = 60  # 60-second segments
    current_time = start_time

    while current_time + timedelta(seconds=duration) <= end_time:
        seg_start = current_time
        seg_end = current_time + timedelta(seconds=duration)
        current_time = seg_end

        # Query
        try:
            data = query_influx_data(
                start_time=seg_start.isoformat(timespec="seconds"),
                end_time=seg_end.isoformat(timespec="seconds"),
                box_id=box_id,
                sensor_id=sensor_id,
                password=password
            )

            key = f"{box_id}_{sensor_id}"
            waveform = data.get(key)
            if waveform is None or waveform.empty:
                print(f"No data for window {seg_start} to {seg_end}")
                continue

            samples = waveform['value'].values
            x = safe_resample(samples, fs_in, fs_out)
            x = preprocess(x, fs_out)

            if 5700 <= len(x) < 6000:
                x = np.pad(x, (0, 6000 - len(x)), mode="constant")

            if len(x) < 6000:
                print(f"Too short after resampling: {len(x)} samples")
                continue

            # PSD windowing
            win_length = int(window_duration * fs_out)
            step = int(win_length * (1 - overlap))
            n_windows = (len(x) - win_length) // step + 1

            if n_windows != 11:
                print(f"Expected 11 PSD windows, got {n_windows}")
                continue

            psd_list = []
            for i in range(n_windows):
                start_idx = i * step
                end_idx = start_idx + win_length
                window_data = x[start_idx:end_idx]
                pxx, _ = welch_psd(window_data, fs_out)
                psd_list.append(pxx)

            psd_array = np.vstack(psd_list)
            psd_vector = psd_array.flatten()

            log_pxx = np.log10(psd_vector + 1e-10)
            z_pxx = (log_pxx - mean) / (std + 1e-6) if mean is not None else log_pxx

            results.append((seg_start, seg_end, z_pxx.astype(np.float32)))

        except Exception as e:
            print(f"Failed to process window {seg_start} to {seg_end}: {e}")
            continue

    return results  # List of (start_time, end_time, psd_vector)

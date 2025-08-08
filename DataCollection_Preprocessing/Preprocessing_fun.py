"""
Waveform Processing Utilities for Infrasound/Earthquake Data

This script provides signal processing functions to prepare raw waveform data
for PSD computation and model inference. It includes DC blocking, filtering,
resampling, and Welch PSD estimation.

Functions:
- dc_block(x, a=0.999):
    Applies a DC blocking filter to remove low-frequency bias.

- preprocess(x, fs):
    Applies DC blocking and a high-pass Butterworth filter with 0.1 Hz cutoff.

- welch_psd(x, fs):
    Computes the Welch Power Spectral Density (PSD) using a 5-second Hann window,
    75% overlap, and keeps frequency components up to 10 Hz.

- safe_resample(x, fs_in, fs_out):
    Resamples the signal from fs_in to fs_out safely by applying low-pass filtering
    before resampling to avoid aliasing.

Intended Use:
- These functions are designed to prepare infrasound waveform data for
  spectral analysis and classification with machine learning models.

Dependencies:
- NumPy
- SciPy (signal module)

Ethan Gelfand, 08/06/2025
"""

import numpy as np
import scipy.signal as signal

## --- Functions for processing waveform data --- ##
def dc_block(x, a=0.999):
    b = [1, -1]
    a_coeffs = [1, -a]
    return signal.filtfilt(b, a_coeffs, x)

def preprocess(x, fs):
    x = dc_block(x)
    low_cutoff = 0.1
    Wn = low_cutoff / (fs / 2)
    b, a = signal.butter(4, Wn, btype='high')
    return signal.filtfilt(b, a, x)

def welch_psd(x, fs):
    window_duration = 5
    nperseg = int(fs * window_duration)
    noverlap = int(nperseg * 0.75)
    nfft = int(2 ** np.ceil(np.log2(nperseg)))

    window = signal.windows.hann(nperseg)
    f, pxx = signal.welch(x, fs, window=window, noverlap=noverlap, nfft=nfft)

    keep = f <= 10
    return pxx[keep], f[keep]

def safe_resample(x, fs_in, fs_out):
    x = dc_block(x)
    fc = 0.9 * min(fs_in, fs_out) / 2
    b_lp, a_lp = signal.butter(4, fc / (fs_in / 2), btype='low')
    x = signal.filtfilt(b_lp, a_lp, x)
    y = signal.resample_poly(x, fs_out, fs_in)
    return y
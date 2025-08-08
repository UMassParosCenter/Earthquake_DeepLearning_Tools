"""
Script: earthquake_cnn_model.py

This script defines a 1D Convolutional Neural Network (CNN) architecture using PyTorch for
binary classification of time-series data, specifically designed for distinguishing between
earthquake and background infrasound signals.

Architecture Overview:
----------------------
- The model expects 1D input signals (e.g., Welch PSD vectors or raw waveforms) of length `input_length`.
- Input shape: (batch_size, input_length)
- The model includes two convolutional blocks followed by fully connected layers.

Main Components:
----------------
1. `ConvBlock`:
   - Conv1D → BatchNorm → ReLU → MaxPool → Dropout
   - Used to extract local features from 1D signals.

2. `EarthquakeCNN`:
   - Two sequential `ConvBlock`s (with kernel sizes 7 and 5).
   - Fully connected layers:
     - Flatten → FC (128) → ReLU → FC (64) → ReLU → FC (2)
   - Dynamically calculates the input size of the first fully connected layer based on
     the input length after convolution and pooling.

Outputs:
--------
- Final output shape: (batch_size, 2)
  - Represents logits for 2 classes: `[earthquake, background]`

Usage:
------
- Instantiate the model with:
    `model = EarthquakeCNN(input_length=<your_input_length>)`
- Feed normalized tensors to `model(input_tensor)` for inference.

Dependencies:
-------------
- torch
- torch.nn

Note:
-----
- Use CrossEntropyLoss for training since the model outputs logits.
- Ensure input tensors are properly normalized and shaped as (batch_size, input_length).

Ethan Gelfand 08/06/2025
"""

import torch
import torch.nn as nn

#--- model definition ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p=4, pool_kernel=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x
        
class EarthquakeCNN(nn.Module):
    def __init__(self, input_length=1133):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=1, out_channels=16, kernel_size=7)
        self.conv2 = ConvBlock(in_channels=16, out_channels=16, kernel_size=5)
        # Dynamically calculate flattened size
        self.flatten_dim = self._get_flattened_size(input_length)
        self.fc1 = nn.Linear(self.flatten_dim, 128) # Adjust input size based on flattened output
        self.fc_hidden = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 output classes: Earthquake and Background
        
    def _get_flattened_size(self, input_length):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            return x.view(1, -1).shape[1]
    
    def forward(self, X):
        X = X.unsqueeze(1)  # Add channel dimension: (batch_size, 1, features)
        X = self.conv1(X) 
        X = self.conv2(X) 
        X = X.view(X.size(0), -1) # Flatten
        X = torch.relu(self.fc1(X)) # (Batch_size, 128)
        X = torch.relu(self.fc_hidden(X))
        X = self.fc2(X) # (Batch_size, 2)
        return X
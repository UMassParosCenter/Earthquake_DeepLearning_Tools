**Earthquake vs. Background Classification Suite**

This repository contains a complete suite of scripts and utilities for **training, evaluating, and deploying** deep learning models to classify seismic Power Spectral Density (PSD) data into **Earthquake** vs **Background** events.

The workflow covers:
1. **Data collection and preprocessing**
2. **PSD computation and export**
3. **Model training** (2D CNN architecture)
4. **Model evaluation** (cross-validation)
5. **Real-time / live stream inference**
6. **Logging & exporting predictions**

---

Folder Structure
----------------

DataCollection_Preprocessing  
- EarthQuakeData.csv  
    - CSV file obtained from the USGS earthquake catalog.  
- Preprocessing_fun.py  
    - Module containing the preprocessing pipeline functions.  
- generateBackgroundData.py  
    - Script that queries InfluxDB for background data and stores it as a dictionary in a pickle file.
    - Add password in this script.  
- usgsEarthquakeDataGrabber.py  
    - Script that queries InfluxDB for earthquake event data and stores it as a dictionary in a pickle file.
    - Add password in this script. 
- PSD_Background_processor.py  
    - Processes background data and outputs a dictionary of PSDs for each window.  
- PSD_Earthquake_processor.py  
    - Processes earthquake event data and outputs a dictionary of PSDs for each window.  
- Exported_Paros_Data  
    - Output folder where all pickle files are stored.  
- Makefile
    - The Makefile automates the preprocessing pipeline for training data. Running make will execute Preprocessing scripts.

Eval  
- cnn_model.py  
    - PyTorch class defining the CNN model.  
- DataQueryUtils.py  
    - Functions for live and range queries from InfluxDB, formatting the data for model evaluation.
    - Add password in this script. 
- Preprocessing_fun.py  
    - Preprocessing pipeline functions.  
- TestModel_DataRange.ipynb  
    - Notebook for evaluating the model on a specific data range.
    - Add password in this script. 
- LiveTestModel.ipynb  
    - Notebook for evaluating the model on continuous live data.  
- LoggedData  
    - Directory storing CSV files of exported predictions.  

ModelTraining  
- cnn_model.py  
    - PyTorch class defining the CNN model.  
- psd_pickle_utils.py  
    - Functions for easily importing PSD pickle files and extracting PSDs as NumPy arrays.  
- LoadData.py  
    - Functions for loading fold data splits to train other models on the same dataset as the original CNN.  
    - Useful for ensemble models where validation is performed on unused data.  
- CNN2D.ipynb  
    - Notebook for training the 2D CNN model.  
- fold_outputs  
    - fold_1  
        - CNNmodel.pth  
        - data.npz  
    - fold_2  
        - CNNmodel.pth  
        - data.npz  
    - fold_3  
        - CNNmodel.pth  
        - data.npz  
    - fold_4  
        - CNNmodel.pth  
        - data.npz  
    - fold_5  
        - CNNmodel.pth  
        - data.npz  


---

**Flow for Training and Testing:**

1. **Generate background and earthquake data:**  
    - Obtain the USGS earthquake event CSV.  
    - Query InfluxDB for background data and earthquake events.  
    - Apply preprocessing and generate PSDs.  

2. **Train the model:**  
    - Run the training script for the model.  
    - Model hyperparameters can be adjusted in the model definition script.  
    - The model dynamically calculates flattened layer lengths, so you do not need to recalculate convolutional output sizes.  
    - Ensure adjustments are made in each model definition file if needed.  

3. **Evaluation:**  
    - For live evaluation on incoming data, run the live evaluation script.  
    - For evaluation on a specific data range, run the data range evaluation script.  

**File run order for training:**
- Run the `Makefile` to execute the preprocessing scripts by typing `make` in the terminal:
  - `generateBackgroundData.py`
  - `usgsEarthquakeDataGrabber.py`
  - `PSD_Background_processor.py`
  - `PSD_Earthquake_processor.py`
- `CNN2D.ipynb`
---

Ethan Gelfand, 8/12/2025

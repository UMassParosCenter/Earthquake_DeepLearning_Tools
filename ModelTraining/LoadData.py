"""
Utility: load_all_folds.py

This utility script provides a function to load cross-validation fold data from saved `.npz` files.
It is useful in the context of evaluating or visualizing model performance across multiple
training folds (e.g., in Stratified K-Fold cross-validation experiments).

Functionality:
--------------
- The function `load_all_folds()` scans a directory (e.g., `fold_outputs/`) for subfolders named
  `fold_0`, `fold_1`, ..., `fold_N`.
- Within each fold folder, it looks for a `data.npz` file containing NumPy arrays for training and
  validation splits, as saved during model training.
- It returns a list of dictionaries, one per fold, each containing arrays such as:
  - `X_train`, `y_train`: Training features and labels.
  - `X_val`, `y_val`: Validation features and labels.

Ethan Gelfand 08/06/2025
"""
import os
import numpy as np

def load_all_folds(folder="fold_outputs", num_folds=None):
    """
    Load all saved fold data from a given parent folder.
    
    Parameters:
        folder (str): Path to the folder containing fold subfolders (e.g., "fold_outputs").
        num_folds (int or None): If set, limits the number of folds to load.

    Returns:
        List[Dict[str, np.ndarray]]: A list of dictionaries, one per fold, containing:
            'X_train', 'y_train', 'X_val', 'y_val', 'preds', 'probs', 'train_index', 'val_index'
    """
    all_fold_data = []

    # Get all fold directories
    fold_dirs = sorted([d for d in os.listdir(folder) if d.startswith("fold_")])
    if num_folds:
        fold_dirs = fold_dirs[:num_folds]

    for fold_dir in fold_dirs:
        data_path = os.path.join(folder, fold_dir, "data.npz")
        if os.path.exists(data_path):
            data = np.load(data_path)
            fold_data = {
                'fold': fold_dir,
                'X_train': data['X_train'],
                'y_train': data['y_train'],
                'X_val': data['X_val'],
                'y_val': data['y_val'],
            }
            all_fold_data.append(fold_data)
        else:
            print(f"[Warning] Missing file: {data_path}")

    return all_fold_data

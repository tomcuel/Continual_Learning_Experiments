# =======================================
# Library Imports
# =======================================
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Optional, Tuple


# =======================================
# Load and Preprocess Data of MNIST
# =======================================
def download_and_preprocess_mnist(
    rng_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Download and preprocess the MNIST dataset

    Parameters
    ----------
    rng_seed : int
        Random seed for reproducibility

    Returns
    -------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Test features
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels

    Usage Example
    --------------
    X_train, X_test, y_train, y_test = download_and_preprocess_mnist()
    """
    print("Downloading MNIST dataset...")
    mnist_data = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist_data["data"]
    y = mnist_data["target"]

    # Normalize
    print("Preprocessing data...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Usage:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng_seed, stratify=y)
    input_dim = int(X_train.shape[1])
    output_dim = int(len(np.unique(y_train)))
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    return X_train, X_test, y_train, y_test, input_dim, output_dim


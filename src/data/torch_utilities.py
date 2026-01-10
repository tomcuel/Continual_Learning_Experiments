# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# =======================================
# Tensor Creation
# =======================================
def df_to_tensor_dataset(
    X: np.ndarray,
    y: np.ndarray
) -> TensorDataset:
    """
    Convert numpy arrays to PyTorch TensorDataset

    Parameters
    ----------
    X : np.ndarray
        Feature data
    y : np.ndarray
        Labels

    Returns
    -------
    TensorDataset
        PyTorch TensorDataset containing features and labels

    Usage Example
    --------------
    dataset = df_to_tensor_dataset(X_train, y_train)
    """
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


def filter_dataset_by_classes(dataset, allowed_classes):
    X, y = dataset.tensors
    mask = torch.isin(y, torch.tensor(allowed_classes))
    return torch.utils.data.TensorDataset(X[mask], y[mask])


# =======================================
# DataLoader Creation
# =======================================
def make_dataloaders(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False
) -> DataLoader:
    """
    Create DataLoader from Dataset
    
    Parameters
    ----------
    dataset : Dataset
        PyTorch Dataset
    batch_size : int
        Batch size for DataLoader
    shuffle : bool
        Whether to shuffle the data

    Returns
    -------
    DataLoader
        PyTorch DataLoader

    Usage Example
    --------------
    dataloader = make_dataloaders(dataset, batch_size=32, shuffle=True)
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =======================================
# Activation function
# =======================================
_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "softmax_logit": None  # handled as logits for CrossEntropy
}


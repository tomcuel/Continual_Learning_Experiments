# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


from src.data.torch_utilities import make_dataloaders


# =======================================
# Prediction & Evaluation for Task-IL
# =======================================
def predict_loader_task_il(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    task_id: int,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict using the model on the given DataLoader for a specific task

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model
    loader : torch.utils.data.DataLoader
        DataLoader for the dataset to predict on
    task_id : int
        Task ID for Task-IL models
    device : Optional[torch.device]
        Device to run the model on (Default: model's device)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        y_pred : np.ndarray
            Predicted class labels
        y_probs : np.ndarray
            Predicted class probabilities
        y_true : np.ndarray
            True class labels

    Usage Example
    --------------
    y_pred, y_probs, y_true = predict_loader_task_il(model, loader, task_id, device)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch, task_id)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    return (torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy(), torch.cat(all_targets).numpy())

def accuracy_score_task_il(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute accuracy score

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_pred : np.ndarray
        Predicted class labels

    Returns
    -------
    float
        Accuracy score

    Usage Example
    --------------
    acc = accuracy_score_task_il(y_true, y_pred)
    """
    return np.mean(y_true == y_pred)

def evaluate_task_task_il(
    model: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    task_id: int,
    device: Optional[torch.device] = None,
    batch_size: int = 64
) -> float:
    """
    Evaluate model accuracy on a specific task dataset

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model
    dataset : torch.utils.data.TensorDataset
        Dataset for the task to evaluate on
    task_id : int
        Task ID for Task-IL models
    device : Optional[torch.device]
        Device to run the model on (Default: model's device)
    batch_size : int
        Batch size for DataLoader

    Returns
    -------
    float
        Accuracy score

    Usage Example
    --------------
    acc = evaluate_task_task_il(model, dataset, task_id, device, batch_size)
    """
    loader = make_dataloaders(dataset, batch_size=batch_size, shuffle=False)
    y_pred, _, y_true = predict_loader_task_il(model, loader, task_id, device)
    return accuracy_score_task_il(y_true, y_pred)


# =======================================
# Prediction & Evaluation (general methods)
# =======================================
def predict_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict using the model on the given DataLoader

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model
    loader : torch.utils.data.DataLoader
        DataLoader for the dataset to predict on
    device : Optional[torch.device]
        Device to run the model on (Default: model's device)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        y_pred : np.ndarray
            Predicted class labels
        y_probs : np.ndarray
            Predicted class probabilities
        y_true : np.ndarray
            True class labels

    Usage Example
    --------------
    y_pred, y_probs, y_true = predict_loader(model, loader, device)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    return (torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy(), torch.cat(all_targets).numpy())

def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute accuracy score

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_pred : np.ndarray
        Predicted class labels

    Returns
    -------
    float
        Accuracy score

    Usage Example
    --------------
    acc = accuracy_score(y_true, y_pred)
    """
    return np.mean(y_true == y_pred)

def evaluate_task(
    model: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    device: Optional[torch.device] = None,
    batch_size: int = 64
) -> float:
    """
    Evaluate model accuracy on a dataset

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model
    dataset : torch.utils.data.TensorDataset
        Dataset to evaluate on
    device : Optional[torch.device]
        Device to run the model on (Default: model's device)
    batch_size : int
        Batch size for DataLoader

    Returns
    -------
    float
        Accuracy score

    Usage Example
    --------------
    acc = evaluate_task(model, dataset, device, batch_size)
    """
    loader = make_dataloaders(dataset, batch_size=batch_size, shuffle=False)
    y_pred, _, y_true = predict_loader(model, loader, device)
    return accuracy_score(y_true, y_pred)


# =======================================
# Prediction & Evaluation for Class-IL and DEN
# =======================================
def predict_loader_den(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    seen_classes: List[int],
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict using the model on the given DataLoader for Class-IL models

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model
    loader : torch.utils.data.DataLoader
        DataLoader for the dataset to predict on
    seen_classes : List[int]
        List of classes seen so far
    device : Optional[torch.device]
        Device to run the model on (Default: model's device)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        y_pred : np.ndarray
            Predicted class labels
        y_true : np.ndarray
            True class labels

    Usage Example
    --------------
    y_pred, y_true = predict_loader_den(model, loader, seen_classes, device
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            # take only seen classes
            task_logits = logits[:, seen_classes]

            # prediction index in task_logits
            preds_task = task_logits.argmax(dim=1)

            # remap back to original labels
            preds = torch.tensor([seen_classes[i] for i in preds_task], device=device)

            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()

def accuracy_score_den(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute accuracy score

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_pred : np.ndarray
        Predicted class labels

    Returns
    -------
    float
        Accuracy score

    Usage Example
    --------------
    acc = accuracy_score_den(y_true, y_pred)
    """
    return np.mean(y_true == y_pred)

def evaluate_task_den(
    model: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    seen_classes: List[int],
    device: Optional[torch.device] = None,
    batch_size: int = 64
) -> float:
    """
    Evaluate model accuracy on a specific task dataset for Class-IL models

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model
    dataset : torch.utils.data.TensorDataset
        Dataset for the task to evaluate on
    seen_classes : List[int]
        List of classes seen so far
    device : Optional[torch.device]
        Device to run the model on (Default: model's device)
    batch_size : int
        Batch size for DataLoader

    Returns
    -------
    float
        Accuracy score

    Usage Example
    --------------
    acc = evaluate_task_den(model, dataset, seen_classes, device, batch_size)
    """
    loader = make_dataloaders(dataset, batch_size=batch_size, shuffle=False)
    y_pred, y_true = predict_loader_den(model, loader, seen_classes, device)
    return accuracy_score_den(y_true, y_pred)


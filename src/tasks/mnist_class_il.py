# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


from src.data.torch_utilities import df_to_tensor_dataset


# =======================================
# Create Class-IL tasks from MNIST
# =======================================
def create_mnist_task_il(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    num_tasks: int = 5,
    is_print: bool = True
) -> List[torch.utils.data.TensorDataset]:
    """
    Create Task-IL tasks from MNIST dataset by splitting digits into separate tasks

    Parameters
    ----------
    X_train : np.ndarray
        Training data of shape (N, 28, 28)
    y_train : np.ndarray
        Training labels of shape (N,)
    num_tasks : int
        Number of tasks
    is_print : bool
        Whether to print task creation info

    Returns
    -------
    List[torch.utils.data.TensorDataset]
        List of TensorDatasets for each task

    Usage Example
    --------------
    tasks = create_mnist_task_il(X_train, y_train)
    """
    if is_print:
        print("Creating Task-IL tasks from MNIST...")
    tasks = []
    digits_per_task = 10 // num_tasks

    for t in range(num_tasks):
        digit_indices = np.where((y_train >= t * digits_per_task) & (y_train < (t + 1) * digits_per_task))[0]
        X_task = X_train[digit_indices]
        y_task = y_train[digit_indices]
        task_dataset = df_to_tensor_dataset(X_task, y_task)
        tasks.append(task_dataset)

    if is_print:
        print(f"Created {len(tasks)} tasks.")
        for i, task in enumerate(tasks):
            print(f" Task {i} of len {len(task)} with labels {task.tensors[1][:10]} ...")
    return tasks


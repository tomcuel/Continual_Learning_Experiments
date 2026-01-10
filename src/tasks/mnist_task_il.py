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
# Create TASK-IL tasks from MNIST
# =======================================
"""
Definition:
- Task ID is known at training and inference
- Each task can have: its own output head or remapped labels
- Easiest scenario
- Forgetting can be avoided almost entirely

Each task:
- Only sees 2 digits
- Labels are remapped to {0,1}
"""
def make_task_il_tasks(
    X: np.ndarray, 
    y: np.ndarray, 
    num_tasks: int = 5,
    is_print: bool = True
) -> List[torch.utils.data.TensorDataset]:
    """
    Create Task-IL tasks from MNIST by splitting digits into separate tasks

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (N, 28*28)
    y : np.ndarray
        Input labels of shape (N,)
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
    tasks = make_task_il_tasks(X, y)
    """
    if is_print:
        print("Creating Task-IL tasks from MNIST...")
    tasks = []
    digits_per_task = 10 // num_tasks

    for t in range(num_tasks):
        digits = list(range(t * digits_per_task, (t + 1) * digits_per_task))
        idx = np.isin(y, digits)
        X_task = X[idx]
        y_task = y[idx]
        label_map = {d: i for i, d in enumerate(digits)}  # Remap labels to {0,1}
        y_task = np.array([label_map[label] for label in y_task])
        tasks.append(df_to_tensor_dataset(X_task, y_task))

    if is_print:
        print(f"Created {len(tasks)} tasks.")
        for i, task in enumerate(tasks):
            print(f" Task {i} of len {len(task)} with labels {task.tensors[1][:10]} ...")
    return tasks


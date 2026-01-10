# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


from src.data.torch_utilities import df_to_tensor_dataset
from src.data.plotting_mnist import plot_mnist_images


# =======================================
# Create DOMAIN-IL tasks from MNIST
# =======================================
"""
Definition:
- Task ID is NOT known at inference
- All tasks share the same output head
- The input distribution changes across tasks (e.g., rotations, noise)
- Forgetting can still occur if the model adapts too much to the new domain
"""
def make_domain_il_tasks(
    X: np.ndarray,
    y: np.ndarray,
    num_tasks: int = 5, 
    is_print: bool = True, 
    is_plot: bool = False
) -> List[torch.utils.data.TensorDataset]:
    """
    Create Domain-IL tasks from MNIST with different drifts for each task

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
    is_plot : bool
        Whether to plot sample images for each task

    Returns
    -------
    List[torch.utils.data.TensorDataset]
        List of TensorDatasets for each task

    Usage Example
    --------------
    tasks = make_domain_il_tasks(X, y)
    """
    if is_print:
        print("Creating Domain-IL tasks from MNIST...")
    tasks = []
    X = X.reshape(-1, 28, 28)  # reshape to images

    # Define a list of possible drifts
    drift_types = ["rotation", "noise", "contrast"]

    for t in range(num_tasks):
        X_task = X.copy()
        # ony take len(X) / num_tasks data by task
        n_samples = len(X) // num_tasks
        X_task = X_task[t * n_samples:(t + 1) * n_samples]
        y_task = y[t * n_samples:(t + 1) * n_samples]

        # Randomly choose a drift type for this task
        drift = drift_types[t % len(drift_types)]  # cycle through types
        if drift == "rotation":
            angle = 15 * t  # e.g., 0, 15, 30, 45, ...
            X_task = np.array([np.rot90(img, k=angle // 90) for img in X_task])
        elif drift == "noise":
            noise_level = 0.1 * (t+1)
            X_task = X_task + noise_level * np.random.randn(*X_task.shape)
            X_task = np.clip(X_task, 0, 1)
        elif drift == "contrast":
            factor = 1 + 0.2 * t
            X_task = np.clip(X_task * factor, 0, 1)

        # Flatten back to (N, 28*28)
        X_task_flat = X_task.reshape(len(X_task), -1)
        print(f"Task {t}: Applied {drift} drift")
        if is_plot:
            plot_mnist_images(X=X_task_flat, y=y_task, num_images=5)
        task_dataset = df_to_tensor_dataset(X_task_flat, y_task)
        tasks.append(task_dataset)

    if is_print:
        print(f"Created {len(tasks)} tasks.")
        for i, task in enumerate(tasks):
            print(f" Task {i} of len {len(task)} with labels {task.tensors[1][:10]} ...")
    return tasks


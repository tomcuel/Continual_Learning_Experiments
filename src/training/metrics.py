# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from typing import Dict, List, Optional, Tuple


# =======================================
# Metrics to track performance
# =======================================
class AccuracyMatrix:
    def __init__(self, 
        num_tasks: int
    ):
        """
        Accuracy Matrix to track performance across tasks

        Parameters
        ----------
        num_tasks : int
            Number of tasks
        """
        self.num_tasks = num_tasks
        self.matrix = np.full((num_tasks, num_tasks), np.nan)

    def update(self,
        task_i: int,
        task_t: int,
        accuracy: float
    ):
        """
        Update accuracy matrix

        Parameters
        ----------
        task_i : int
            Task index being evaluated
        task_t : int
            Task index after which evaluation is done
        accuracy : float
            Accuracy value to record

        Usage Example
        --------------
        acc_matrix.update(task_i=0, task_t=1, accuracy=0.85)
        """
        self.matrix[task_i, task_t] = accuracy

    def get(self) -> np.ndarray:
        """
        Get the accuracy matrix

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Accuracy matrix of shape (num_tasks, num_tasks)

        Usage Example
        --------------
        A = acc_matrix.get()
        """
        return self.matrix

def average_accuracy(
    A: np.ndarray
) -> float:
    """
    Compute Average Accuracy (AA): measures average accuracy at the end of training

    Parameters
    ----------
    A : np.ndarray
        Accuracy matrix of shape (T, T)

    Returns
    -------
    float
        Average accuracy

    Usage Example
    --------------
    avg_acc = average_accuracy(A)
    """
    T = A.shape[0]
    return np.nanmean([A[i, T-1] for i in range(T)])

def forgetting(
    A: np.ndarray
) -> float:
    """
    Compute Forgetting (F): measures worst performance drop per task

    Parameters
    ----------
    A : np.ndarray
        Accuracy matrix of shape (T, T)

    Returns
    -------
    float
        Forgetting measure

    Usage Example
    --------------
    fgt = forgetting(A)
    """
    T = A.shape[0]
    forgettings = []

    for i in range(T - 1):
        best_past = np.nanmax(A[i, :T-1])
        final = A[i, T-1]
        forgettings.append(best_past - final)

    return np.mean(forgettings)

def forward_transfer(
    A: np.ndarray,
    baseline: List[float]
) -> float:
    """
    Compute Forward Transfer (FT): measures influence of learning previous tasks on new tasks

    Parameters
    ----------
    A : np.ndarray
        Accuracy matrix of shape (T, T)
    baseline : List[float]
        Baseline accuracies for each task before training

    Returns
    -------
    float
        Forward transfer measure

    Usage Example
    --------------
    ft = forward_transfer(A, baseline)
    """
    T = A.shape[0]
    fts = []

    for i in range(1, T):
        fts.append(A[i, i-1] - baseline[i])

    return np.mean(fts)

def backward_transfer(
    A: np.ndarray
) -> float:
    """
    Compute Backward Transfer (BT): measures influence of learning new tasks on previous tasks

    Parameters
    ----------
    A : np.ndarray
        Accuracy matrix of shape (T, T)

    Returns
    -------
    float
        Backward transfer measure

    Usage Example
    --------------
    bt = backward_transfer(A)
    """
    T = A.shape[0]
    bts = []

    for i in range(T - 1):
        bts.append(A[i, T-1] - A[i, i])

    return np.mean(bts)


# =======================================
# Final Evaluation Metrics
# =======================================
def evaluate_accuracy_matrix(
    acc_matrix: AccuracyMatrix
):
    """
    Evaluate and print final metrics from accuracy matrix

    Parameters
    ----------
    acc_matrix : AccuracyMatrix
        AccuracyMatrix object containing performance data

    Returns
    -------
    None

    Usage Example
    --------------
    evaluate_accuracy_matrix(acc_matrix)
    """
    A = acc_matrix.get()
    print("Final Evaluation Metrics:")
    print("Accuracy Matrix:\n", np.round(A, 2))
    print("Average Accuracy:", average_accuracy(A))
    print("Forgetting:", forgetting(A))
    print("Backward Transfer:", backward_transfer(A))  
    # Note: Forward Transfer requires baseline accuracies
    print("Forward Transfer: N/A (requires baseline)")


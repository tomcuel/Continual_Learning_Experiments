import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, List, Optional, Tuple


from src.training.metrics import AccuracyMatrix, evaluate_accuracy_matrix


if __name__ == "__main__":
    # Example usage of AccuracyMatrix
    num_tasks = 3
    acc_matrix = AccuracyMatrix(num_tasks=num_tasks)

    # Simulate updating the accuracy matrix
    acc_matrix.update(task_i=0, task_t=0, accuracy=0.9)
    acc_matrix.update(task_i=0, task_t=1, accuracy=0.85)
    acc_matrix.update(task_i=0, task_t=2, accuracy=0.75)
    acc_matrix.update(task_i=1, task_t=1, accuracy=0.88)
    acc_matrix.update(task_i=1, task_t=2, accuracy=0.8)
    acc_matrix.update(task_i=2, task_t=2, accuracy=0.95)

    # Retrieve and print the accuracy matrix
    matrix = acc_matrix.get()
    print("Accuracy Matrix:")
    print(matrix)

    # Evaluate overall performance
    evaluate_accuracy_matrix(acc_matrix)


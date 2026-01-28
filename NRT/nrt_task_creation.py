import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, List, Optional, Tuple


from src.data.download_mnist import download_and_preprocess_mnist
from src.tasks.mnist_class_il import create_mnist_task_il
from src.tasks.mnist_domain_il import make_domain_il_tasks
from src.tasks.mnist_task_il import make_task_il_tasks


if __name__ == "__main__":
    # Download and preprocess MNIST data
    X_train, X_test, y_train, y_test, input_dim, output_dim = download_and_preprocess_mnist()

    # Create Class-IL tasks
    class_il_tasks = create_mnist_task_il(X_train, y_train, num_tasks=5, is_print=True)

    # Create Domain-IL tasks
    domain_il_tasks = make_domain_il_tasks(X_train, y_train, num_tasks=5, is_print=True)

    # Create Task-IL tasks
    task_il_tasks = make_task_il_tasks(X_train, y_train, num_tasks=5, is_print=True)


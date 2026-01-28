import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from typing import Dict, List, Optional, Tuple


from src.data.download_mnist import download_and_preprocess_mnist
from src.tasks.mnist_class_il import create_mnist_task_il
from src.tasks.mnist_domain_il import make_domain_il_tasks
from src.tasks.mnist_task_il import make_task_il_tasks
from src.methods.replaybuffer import ReplayBuffer_ClassIL, ReplayBuffer_DomainIL, ReplayBuffer_TaskIL


if __name__ == "__main__":
    # Download and preprocess MNIST data
    X_train, X_test, y_train, y_test, input_dim, output_dim = download_and_preprocess_mnist()

    # Create Class-IL tasks and initialize replay buffer
    class_il_tasks = create_mnist_task_il(X_train, y_train, num_tasks=5, is_print=True)
    class_il_replay_buffer = ReplayBuffer_ClassIL(max_size_per_class=100, num_classes=10)
    print("Initialized Class-IL Replay Buffer of length for the first task:", len(class_il_replay_buffer.buffer[0]))
    class_il_replay_buffer.add(X=torch.tensor(X_train[:10], dtype=torch.float32), y=torch.tensor(y_train[:10], dtype=torch.long))
    print("Added samples to Class-IL Replay Buffer of length for the first task:", len(class_il_replay_buffer.buffer[0]))

    # Create Domain-IL tasks and initialize replay buffer
    domain_il_tasks = make_domain_il_tasks(X_train, y_train, num_tasks=5, is_print=True)
    domain_il_replay_buffer = ReplayBuffer_DomainIL(capacity=500)
    print("Initialized Domain-IL Replay Buffer of length:", len(domain_il_replay_buffer.memory))
    domain_il_replay_buffer.add(torch.utils.data.TensorDataset(torch.tensor(X_train[:10], dtype=torch.float32), torch.tensor(y_train[:10], dtype=torch.long)))
    print("Added samples to Domain-IL Replay Buffer of length:", len(domain_il_replay_buffer.memory))

    # Create Task-IL tasks and initialize replay buffer
    task_il_tasks = make_task_il_tasks(X_train, y_train, num_tasks=5, is_print=True)
    task_il_replay_buffer = ReplayBuffer_TaskIL(max_size_per_task=500)
    task_il_replay_buffer.buffer[0] = []  # Initialize buffer for the first task
    print("Initialized Task-IL Replay Buffer of length for the first task:", len(task_il_replay_buffer.buffer[0]))
    task_il_replay_buffer.add(task_id=0, X=torch.tensor(X_train[:10], dtype=torch.float32), y=torch.tensor(y_train[:10], dtype=torch.long))
    print("Added samples to Task-IL Replay Buffer of length for the first task:", len(task_il_replay_buffer.buffer[0]))


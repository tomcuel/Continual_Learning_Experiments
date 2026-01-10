# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


# =======================================
# Replay Buffer Implementation for Class-IL
# =======================================
class ReplayBuffer_ClassIL:
    def __init__(self, 
        max_size_per_class: int = 100, 
        num_classes: int = 10
    ):
        """
        Simple Replay Buffer for Class-IL

        Parameters
        ----------
        max_size_per_class : int
            Maximum number of samples to store per class
        num_classes : int
            Number of classes
        """

        self.max_size = max_size_per_class
        self.num_classes = num_classes
        self.buffer = {c: [] for c in range(num_classes)}

    def add(self, 
        X: torch.Tensor,
        y: torch.Tensor
    ):
        """
        Add samples to the replay buffer
        
        Parameters
        ----------
        X : torch.Tensor
            Input samples (N, D)
        y : torch.Tensor
            Corresponding labels (N,)

        Returns
        -------
        None

        Usage Example
        --------------
        replay_buffer.add(X_batch, y_batch)
        """
        for xi, yi in zip(X, y):
            c = int(yi)
            if len(self.buffer[c]) < self.max_size:
                self.buffer[c].append(xi)
            else:
                # Reservoir sampling: randomly replace existing sample
                idx = np.random.randint(0, self.max_size)
                self.buffer[c][idx] = xi

    def sample(self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the replay buffer

        Parameters
        ----------
        batch_size : int
            Number of samples to draw

        Returns
        -------
        X_batch : torch.Tensor
            Sampled input batch
        y_batch : torch.Tensor
            Sampled labels batch

        Usage Example
        --------------
        X_batch, y_batch = replay_buffer.sample(batch_size)
        """
        # Sample equally from all classes
        X_batch, y_batch = [], []
        classes = list(self.buffer.keys())
        while len(X_batch) < batch_size:
            c = np.random.choice(classes)
            if len(self.buffer[c]) > 0:
                idx = np.random.randint(len(self.buffer[c]))
                X_batch.append(self.buffer[c][idx])
                y_batch.append(c)
        X_batch = torch.stack(X_batch)
        y_batch = torch.tensor(y_batch)
        return X_batch, y_batch

    def __len__(self):
        """
        Get the total number of samples in the buffer
        """
        return sum(len(v) for v in self.buffer.values())


# =======================================
# Replay Buffer Implementation for Domain-IL
# =======================================
class ReplayBuffer_DomainIL:
    def __init__(self, 
        capacity: int = 1000
    ):
        """
        Simple Replay Buffer for Domain-IL

        Parameters
        ----------
        capacity : int
            Maximum number of samples to store in the buffer
        """
        self.capacity = capacity
        self.memory = []

    def add(self, 
        dataset: torch.utils.data.Dataset
    ):
        """
        Add samples from dataset to the replay buffer

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to add samples from

        Returns
        -------
        None

        Usage Example
        --------------
        replay_buffer.add(dataset)
        """
        for x, y in dataset:
            if len(self.memory) >= self.capacity:
                self.memory.pop(0)
            self.memory.append((x.cpu(), y.cpu()))

    def sample(self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the replay buffer

        Parameters
        ----------
        batch_size : int
            Number of samples to draw

        Returns
        -------
        X_batch : torch.Tensor
            Sampled input batch
        y_batch : torch.Tensor
            Sampled labels batch

        Usage Example
        --------------
        X_batch, y_batch = replay_buffer.sample(batch_size)
        """
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        X, y = zip(*batch)
        return torch.stack(X), torch.tensor(y)

    def __len__(self):
        """
        Get the total number of samples in the buffer
        """
        return len(self.memory)
    

# =======================================
# Replay Buffer Implementation for Task-IL
# =======================================
class ReplayBuffer_TaskIL:
    def __init__(self, 
        max_size_per_task: int
    ):
        """
        Simple Replay Buffer for Task-IL

        Parameters
        ----------
        max_size_per_task : int
            Maximum number of samples to store per task
        """
        self.max_size = max_size_per_task
        self.buffer = {}  # task_id -> list[(x, y)]

    def add(self, 
        task_id: int, 
        X: torch.Tensor, 
        y: torch.Tensor
    ):  
        """
        Add samples to the replay buffer for a specific task

        Parameters
        ----------
        task_id : int
            Identifier for the task
        X : torch.Tensor
            Input samples (N, D)
        y : torch.Tensor
            Corresponding labels (N,)

        Returns
        -------
        None

        Usage Example
        --------------
        replay_buffer.add(task_id, X_batch, y_batch)
        """
        if task_id not in self.buffer:
            self.buffer[task_id] = []

        for xi, yi in zip(X, y):
            if len(self.buffer[task_id]) < self.max_size:
                self.buffer[task_id].append((xi.cpu(), yi.cpu()))
            else:
                idx = np.random.randint(self.max_size)
                self.buffer[task_id][idx] = (xi.cpu(), yi.cpu())

    def sample(self, 
        batch_size: int, 
        task_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replay only from the current task

        Parameters
        ----------
        batch_size : int
            Number of samples to draw
        task_id : int
            Identifier for the task to sample from

        Returns
        -------
        X_batch : torch.Tensor
            Sampled input batch
        y_batch : torch.Tensor
            Sampled labels batch

        Usage Example
        --------------
        X_batch, y_batch = replay_buffer.sample(batch_size, task_id)
        """
        memory = self.buffer.get(task_id, [])
        indices = np.random.choice(len(memory), batch_size, replace=False)
        X, y = zip(*[memory[i] for i in indices])
        return torch.stack(X), torch.tensor(y)

    def sample_all_tasks(self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optional: replay from all previous tasks

        Parameters
        ----------
        batch_size : int
            Number of samples to draw

        Returns
        -------
        X_batch : torch.Tensor
            Sampled input batch
        y_batch : torch.Tensor
            Sampled labels batch

        Usage Example
        --------------
        X_batch, y_batch = replay_buffer.sample_all_tasks(batch_size)
        """
        all_samples = []
        for mem in self.buffer.values():
            all_samples.extend(mem)

        indices = np.random.choice(len(all_samples), batch_size, replace=False)
        X, y = zip(*[all_samples[i] for i in indices])
        return torch.stack(X), torch.tensor(y)
    
    
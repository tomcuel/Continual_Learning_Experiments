# ===============================
# Library Imports
# ===============================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple


from src.data.torch_utilities import make_dataloaders
from src.methods.ewc import EWC
from src.methods.replaybuffer import ReplayBuffer_TaskIL, ReplayBuffer_DomainIL, ReplayBuffer_ClassIL
from src.models.den import DEN, class_il_loss
from src.models.mlp_tasks import DeepNN_TaskIL
from src.models.mlp import DeepNN
from src.models.prog_net import ProgressiveNet


# =======================================
# Training Loop for Task-IL (with optional Replay and EWC)
# =======================================
def train_mlp_on_task_task_il(
    model: DeepNN_TaskIL,
    device: torch.device,
    train_dataset: torch.utils.data.TensorDataset,
    task_id: int,
    batch_size: int = 64,
    epochs: int = 10,
    gamma: float = 0.5,
    learning_rate: float = 0.001,
    loss_function: str = "cross_entropy",
    step_size: int = 20,
    weight_decay: float = 0.00001,
    replay_buffer: Optional[ReplayBuffer_TaskIL] = None, 
    replay_weight: Optional[float] = 1.0,
    ewc: Optional[EWC] = None,
    print_every: Optional[int] = 1
) -> DeepNN_TaskIL:
    """
    Training loop for Task-IL models with optional Replay and EWC

    Parameters
    ----------
    model : DeepNN_TaskIL
        The neural network model
    device : torch.device
        Device to run the model on
    train_dataset : torch.utils.data.TensorDataset
        Training dataset for the current task
    task_id : int
        Task ID for Task-IL models
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    gamma : float
        Learning rate decay factor
    learning_rate : float
        Initial learning rate
    loss_function : str
        Loss function to use ("cross_entropy" or "mse")
    step_size : int
        Step size for learning rate scheduler
    weight_decay : float
        Weight decay for optimizer
    replay_buffer : Optional[ReplayBuffer_TaskIL]
        Replay buffer for Task-IL (Default: None)
    replay_weight : Optional[float]
        Weight for replay loss (Default: 1.0)
    ewc : Optional[EWC]
        EWC object for regularization (Default: None)
    print_every : Optional[int]
        Frequency of printing training progress (Default: 1)

    Returns
    -------
    model : DeepNN_TaskIL
        Trained model

    Usage Example
    --------------
    model = train_mlp_on_task_task_il(model, device, train_dataset, task_id)
    """
    train_loader = make_dataloaders(train_dataset, batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss() if loss_function == "cross_entropy" else nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Current task loss
            outputs = model(X_batch, task_id)
            loss = criterion(outputs, y_batch)

            # Replay loss
            if replay_buffer is not None and len(replay_buffer.buffer) >= batch_size:
                X_rep, y_rep = replay_buffer.sample(batch_size)
                X_rep, y_rep = X_rep.to(device), y_rep.to(device)
                logits_rep = model(X_rep)
                loss_rep = criterion(logits_rep, y_rep)
                loss += replay_weight * loss_rep

            # EWC penalty
            if ewc is not None:
                loss += ewc.penalty(model)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader.dataset)

        # Print progress
        if (epoch+1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}")

    return model


# =======================================
# Training Loop (with optional Replay and EWC)
# =======================================
def train_mlp_on_task(
    model: DeepNN,
    device: torch.device,
    train_dataset: torch.utils.data.TensorDataset,
    batch_size: int = 64,
    epochs: int = 10,
    gamma: float = 0.5,
    learning_rate: float = 0.001,
    loss_function: str = "cross_entropy",
    step_size: int = 20,
    weight_decay: float = 0.00001,
    replay_buffer_class: Optional[ReplayBuffer_ClassIL] = None,
    replay_buffer_domain: Optional[ReplayBuffer_DomainIL] = None,
    replay_weight: Optional[float] = 1.0,
    ewc: Optional[EWC] = None,
    print_every: Optional[int] = 1
) -> DeepNN:
    """
    Training loop for Domain-IL and Class-IL models with optional Replay and EWC

    Parameters
    ----------
    model : DeepNN
        The neural network model
    device : torch.device
        Device to run the model on
    train_dataset : torch.utils.data.TensorDataset
        Training dataset for the current task
    task_id : int
        Task ID for Task-IL models
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    gamma : float
        Learning rate decay factor
    learning_rate : float
        Initial learning rate
    loss_function : str
        Loss function to use ("cross_entropy" or "mse")
    step_size : int
        Step size for learning rate scheduler
    weight_decay : float
        Weight decay for optimizer
    replay_buffer_class : Optional[ReplayBuffer_ClassIL]
        Replay buffer for Class-IL (Default: None)
    replay_buffer_domain : Optional[ReplayBuffer_DomainIL]
        Replay buffer for Domain-IL (Default: None)
    replay_weight : Optional[float]
        Weight for replay loss (Default: 1.0)
    ewc : Optional[EWC]
        EWC object for regularization (Default: None)
    print_every : Optional[int]
        Frequency of printing training progress (Default: 1)

    Returns
    -------
    model : DeepNN
        Trained model

    Usage Example
    --------------
    model = train_mlp_on_task(model, device, train_dataset, task_id)
    """
    train_loader = make_dataloaders(train_dataset, batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss() if loss_function == "cross_entropy" else nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Current task loss
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Replay loss (Class-IL or Domain-IL depending on which buffer is provided)
            if replay_buffer_class is not None and len(replay_buffer_class.buffer) >= batch_size:
                X_rep, y_rep = replay_buffer_class.sample(batch_size)
                X_rep, y_rep = X_rep.to(device), y_rep.to(device)
                logits_rep = model(X_rep)
                loss_rep = criterion(logits_rep, y_rep)
                loss += replay_weight * loss_rep
            elif replay_buffer_domain is not None and len(replay_buffer_domain.memory) >= batch_size:
                X_rep, y_rep = replay_buffer_domain.sample(batch_size)
                X_rep, y_rep = X_rep.to(device), y_rep.to(device)
                logits_rep = model(X_rep)
                loss_rep = criterion(logits_rep, y_rep)
                loss += replay_weight * loss_rep

            # EWC penalty
            if ewc is not None:
                loss += ewc.penalty(model)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader.dataset)

        # Print progress
        if (epoch+1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}")

    return model


# =======================================
# Training Loop for Task-IL for Progressive Nets (with optional Replay and EWC)
# =======================================
def train_prog_net_on_task_task_il(
    model: ProgressiveNet, 
    device: torch.device,
    train_dataset: torch.utils.data.TensorDataset,
    task_id: int,
    batch_size: int = 64,
    epochs: int = 10,
    gamma: float = 0.5,
    learning_rate: float = 0.001,
    loss_function: str = "cross_entropy",
    step_size: int = 20,
    weight_decay: float = 0.00001,
    replay_buffer: Optional[ReplayBuffer_TaskIL] = None, 
    replay_weight: Optional[float] = 1.0,
    ewc: Optional[EWC] = None,
    print_every: Optional[int] = 1
) -> ProgressiveNet:
    """
    Training loop for Task-IL Progressive Nets with optional Replay and EWC

    Parameters
    ----------
    model : ProgressiveNet
        The Progressive Neural Network model
    device : torch.device
        Device to run the model on
    train_dataset : torch.utils.data.TensorDataset
        Training dataset for the current task
    task_id : int
        Task ID for Task-IL models
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    gamma : float
        Learning rate decay factor
    learning_rate : float
        Initial learning rate
    loss_function : str
        Loss function to use ("cross_entropy" or "mse")
    step_size : int
        Step size for learning rate scheduler
    weight_decay : float
        Weight decay for optimizer
    replay_buffer : Optional[ReplayBuffer_TaskIL]
        Replay buffer for Task-IL (Default: None)
    replay_weight : Optional[float]
        Weight for replay loss (Default: 1.0)
    ewc : Optional[EWC]
        EWC object for regularization (Default: None)
    print_every : Optional[int]
        Frequency of printing training progress (Default: 1)

    Returns
    -------
    model : ProgressiveNet
        Trained model

    Usage Example
    --------------
    model = train_prog_net_on_task_task_il(model, device, train_dataset, task_id)
    """
    train_loader = make_dataloaders(train_dataset, batch_size, shuffle=True)
    
    optimizer = optim.AdamW(list(model.columns[task_id].parameters()) + list(model.heads[task_id].parameters()), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss() if loss_function == "cross_entropy" else nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Current task loss
            outputs = model(X_batch, task_id)
            loss = criterion(outputs, y_batch)

            # Replay loss
            if replay_buffer is not None and len(replay_buffer.buffer) >= batch_size:
                X_rep, y_rep = replay_buffer.sample(batch_size)
                X_rep, y_rep = X_rep.to(device), y_rep.to(device)
                logits_rep = model(X_rep, task_id)
                loss_rep = criterion(logits_rep, y_rep)
                loss += replay_weight * loss_rep

            # EWC penalty
            if ewc is not None:
                loss += ewc.penalty(model)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader.dataset)

        # Print progress
        if (epoch+1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}") 

    return model


# =======================================
# Training Loop for Class-IL for Dynamic Expandable Network (DEN) (with optional Replay and EWC)
# =======================================
def train_den_on_task(
    model: DEN,
    device: torch.device,
    train_dataset: torch.utils.data.TensorDataset,
    task_classes: List[int],
    batch_size: int = 64,
    epochs: int = 10,
    gamma: float = 0.5,
    learning_rate: float = 0.001,
    step_size: int = 20,
    weight_decay: float = 0.00001,
    lambda_sparse: float = 0.001,
    max_grad_norm: float = 1.0,
    replay_buffer: Optional[ReplayBuffer_ClassIL] = None,
    replay_weight: Optional[float] = 1.0,
    ewc: Optional[EWC] = None,
    print_every: Optional[int] = 1
) -> DEN:
    """
    Training loop for Class-IL Dynamic Expandable Network (DEN) with optional Replay and EWC

    Parameters
    ----------
    model : DEN
        The DEN model
    device : torch.device
        Device to run the model on
    train_dataset : torch.utils.data.TensorDataset
        Training dataset for the current task   
    task_classes : List[int]
        List of class indices relevant to the current task
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    gamma : float
        Learning rate decay factor  
    learning_rate : float
        Initial learning rate
    step_size : int
        Step size for learning rate scheduler
    weight_decay : float
        Weight decay for optimizer
    lambda_sparse : float
        Weight for group sparsity regularization
    max_grad_norm : float
        Maximum gradient norm for clipping
    replay_buffer : Optional[ReplayBuffer_ClassIL]
        Replay buffer for Class-IL (Default: None)
    replay_weight : Optional[float]
        Weight for replay loss (Default: 1.0)
    ewc : Optional[EWC]
        EWC object for regularization (Default: None)
    print_every : Optional[int]
        Frequency of printing training progress (Default: 1)

    Returns
    -------
    model : DEN
        Trained model

    Usage Example
    --------------
    model = train_den_on_task(model, device, train_dataset, task_classes)
    """
    train_loader = make_dataloaders(train_dataset, batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            logits = model(X_batch)
            loss = class_il_loss(logits, y_batch, task_classes)
            # group sparsity
            loss += lambda_sparse * sum(layer.weight[~layer.frozen_mask].norm(p=2, dim=1).sum() for layer in model.layers)
            
            # Replay loss
            if replay_buffer is not None and len(replay_buffer.buffer) >= batch_size:
                X_rep, y_rep = replay_buffer.sample(batch_size)
                X_rep, y_rep = X_rep.to(device), y_rep.to(device)
                logits_rep = model(X_rep)
                loss_rep = class_il_loss(logits_rep, y_rep, task_classes)
                loss += replay_weight * loss_rep

            # EWC penalty
            if ewc is not None:
                loss += ewc.penalty(model)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader.dataset)

        # Print progress
        if (epoch+1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}") 

    return model


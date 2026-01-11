# ===============================
# Library Imports
# ===============================
import numpy as np
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple


from src.data.torch_utilities import make_dataloaders, df_to_tensor_dataset
from src.methods.ewc import EWC, EWC_TaskIL
from src.methods.replaybuffer import ReplayBuffer_TaskIL, ReplayBuffer_DomainIL, ReplayBuffer_ClassIL
from src.models.den import DEN, needs_expansion, expand_den, freeze_neurons
from src.models.mlp_tasks import DeepNN_TaskIL
from src.models.mlp import DeepNN
from src.models.prog_net import ProgressiveNet
from src.training.evaluator import evaluate_task_task_il, evaluate_task, evaluate_task_den
from src.training.metrics import AccuracyMatrix
from src.training.task_training import train_mlp_on_task_task_il, train_mlp_on_task, train_prog_net_on_task_task_il, train_den_on_task


# ===============================
# Full Training Pipeline on MLP and for Task-IL
# ===============================
def full_training_task_il(
    model: DeepNN_TaskIL, 
    device: torch.device,
    tasks: List[torch.utils.data.TensorDataset],
    replay_buffer: Optional[ReplayBuffer_TaskIL] = None,
    replay_weight: float = 1.0,
    is_ewc: bool = False,
    lambda_ewc: float = 0.4,
    batch_size: int = 64,
    epochs: int = 10,
    gamma: float = 0.5,
    learning_rate: float = 0.001,
    loss_function: str = "cross_entropy",
    step_size: int = 20,
    weight_decay: float = 0.00001,
    print_every: int = 1
) -> Tuple[DeepNN_TaskIL, AccuracyMatrix]:
    """
    Full training pipeline for Task-IL scenario using MLP with optional Replay and EWC

    Parameters
    ----------
    model : DeepNN_TaskIL
        The MLP model for Task-IL
    device : torch.device
        Device to run the model on
    tasks : List[torch.utils.data.TensorDataset]
        List of tasks as TensorDatasets
    replay_buffer : Optional[ReplayBuffer_TaskIL]
        Replay buffer for Task-IL (Default: None)
    replay_weight : float
        Weight for replay loss (Default: 1.0)
    is_ewc : bool
        Whether to use EWC regularization (Default: False)
    lambda_ewc : float
        EWC regularization strength (Default: 0.4)
    batch_size : int
        Batch size for training (Default: 64)
    epochs : int
        Number of epochs per task (Default: 10)
    gamma : float
        Learning rate decay factor (Default: 0.5)
    learning_rate : float
        Initial learning rate (Default: 0.001)
    loss_function : str
        Loss function to use (Default: "cross_entropy")
    step_size : int
        Step size for learning rate scheduler (Default: 20)
    weight_decay : float
        Weight decay for optimizer (Default: 0.00001)
    print_every : int
        Print progress every n epochs (Default: 1)  

    Returns
    -------
    Tuple[DeepNN_TaskIL, AccuracyMatrix]
        Trained model and accuracy matrix across tasks  

    Usage Example
    --------------
    model, accuracy_matrix = full_training_task_il(model, device, tasks)
    """
    num_tasks = len(tasks)
    accuracy_matrix = AccuracyMatrix(num_tasks=num_tasks)
    ewc = None
    
    for t, task in enumerate(tasks):
        print(f"Training on Task {t} with {len(task)} samples")

        # Train on current task with replay buffer and ewc if provided
        model = train_mlp_on_task_task_il(
            model=model,
            device=device,
            train_dataset=task,
            task_id=t, 
            batch_size=batch_size,
            epochs=epochs,
            gamma=gamma,
            learning_rate=learning_rate,
            loss_function=loss_function,
            step_size=step_size,
            weight_decay=weight_decay,
            replay_buffer=replay_buffer,
            replay_weight=replay_weight,
            ewc=ewc,
            print_every=print_every
        )

        # Evaluate on all seen tasks
        for i in range(t + 1):
            acc = evaluate_task_task_il(
                model=model, 
                dataset=tasks[i],
                task_id=i,
                device=device,
                batch_size=batch_size
            )
            accuracy_matrix.update(task_i=i, task_t=t, accuracy=acc)
            print(f"  Eval on Task {i}: Accuracy = {acc*100:.2f}%")

        # Update replay buffer after task
        if replay_buffer is not None:
            replay_buffer.add(task_id=t, X=task.tensors[0], y=task.tensors[1])

        # Update EWC after task
        if is_ewc:
            task_loader = make_dataloaders(dataset=task, batch_size=batch_size, shuffle=True)
            ewc = EWC_TaskIL(model=model, dataloader=task_loader, task_id=t, device=device, lambda_ewc=lambda_ewc)

    return model, accuracy_matrix


# ===============================
# Full Training Pipeline on MLP and for Domain-IL and Class-IL
# ===============================
def full_training(
    model: DeepNN, 
    device: torch.device,
    tasks: List[torch.utils.data.TensorDataset],
    replay_buffer_class: Optional[ReplayBuffer_ClassIL] = None,
    replay_buffer_domain: Optional[ReplayBuffer_DomainIL] = None,
    replay_weight: float = 1.0,
    is_ewc: bool = False,
    lambda_ewc: float = 0.4,
    batch_size: int = 64,
    epochs: int = 10,
    gamma: float = 0.5,
    learning_rate: float = 0.001,
    loss_function: str = "cross_entropy",
    step_size: int = 20,
    weight_decay: float = 0.00001,
    print_every: int = 1
) -> Tuple[DeepNN, AccuracyMatrix]:
    """
    Full training pipeline for Domain-IL and Class-IL scenarios using MLP with optional Replay and EWC

    Parameters
    ----------
    model : DeepNN
        The MLP model
    device : torch.device
        Device to run the model on
    tasks : List[torch.utils.data.TensorDataset]
        List of tasks as TensorDatasets
    replay_buffer_class : Optional[ReplayBuffer_ClassIL]
        Replay buffer for Class-IL (Default: None)
    replay_buffer_domain : Optional[ReplayBuffer_DomainIL]
        Replay buffer for Domain-IL (Default: None)
    replay_weight : float
        Weight for replay loss (Default: 1.0)
    is_ewc : bool
        Whether to use EWC regularization (Default: False)
    lambda_ewc : float
        EWC regularization strength (Default: 0.4)
    batch_size : int
        Batch size for training (Default: 64)
    epochs : int
        Number of epochs per task (Default: 10)
    gamma : float
        Learning rate decay factor (Default: 0.5)
    learning_rate : float
        Initial learning rate (Default: 0.001)
    loss_function : str
        Loss function to use (Default: "cross_entropy")
    step_size : int
        Step size for learning rate scheduler (Default: 20)
    weight_decay : float
        Weight decay for optimizer (Default: 0.00001)
    print_every : int
        Print progress every n epochs (Default: 1)

    Returns
    -------
    Tuple[DeepNN, AccuracyMatrix]
        Trained model and accuracy matrix across tasks

    Usage Example
    --------------
    model, accuracy_matrix = full_training(model, device, tasks)
    """
    num_tasks = len(tasks)
    accuracy_matrix = AccuracyMatrix(num_tasks=num_tasks)
    ewc = None
    
    for t, task in enumerate(tasks):
        print(f"Training on Task {t} with {len(task)} samples")

        # Train on current task with replay buffer and ewc if provided
        model = train_mlp_on_task(
            model=model,
            device=device,
            train_dataset=task,
            batch_size=batch_size,
            epochs=epochs,
            gamma=gamma,
            learning_rate=learning_rate,
            loss_function=loss_function,
            step_size=step_size,
            weight_decay=weight_decay,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_domain=replay_buffer_domain,
            replay_weight=replay_weight,
            ewc=ewc,
            print_every=print_every
        )

        # Evaluate on all seen tasks
        for i in range(t + 1):
            acc = evaluate_task(
                model=model, 
                dataset=tasks[i],
                device=device,
                batch_size=batch_size
            )
            accuracy_matrix.update(task_i=i, task_t=t, accuracy=acc)
            print(f"  Eval on Task {i}: Accuracy = {acc*100:.2f}%")

        # Update replay buffer after task
        if replay_buffer_class is not None:
            replay_buffer_class.add(X=task.tensors[0], y=task.tensors[1])
        elif replay_buffer_domain is not None:
            replay_buffer_domain.add(dataset=task)

        # Update EWC after task
        if is_ewc:
            task_loader = make_dataloaders(dataset=task, batch_size=batch_size, shuffle=True)
            ewc = EWC(model=model, dataloader=task_loader, device=device, lambda_ewc=lambda_ewc)

    return model, accuracy_matrix


# ===============================
# Full Training Pipeline on Progressive Net and for Task-IL
# ===============================
def full_training_prog_net_task_il(
    model: ProgressiveNet, 
    device: torch.device,
    tasks: List[torch.utils.data.TensorDataset],
    replay_buffer: Optional[ReplayBuffer_TaskIL] = None,
    replay_weight: float = 1.0,
    is_ewc: bool = False,
    lambda_ewc: float = 0.4,
    batch_size: int = 64,
    epochs: int = 10,
    gamma: float = 0.5,
    learning_rate: float = 0.001,
    loss_function: str = "cross_entropy",
    step_size: int = 20,
    weight_decay: float = 0.00001,
    print_every: int = 1
) -> Tuple[ProgressiveNet, AccuracyMatrix]:
    """
    Full training pipeline for Task-IL scenario using Progressive Net with optional Replay and EWC

    Parameters
    ----------
    model : ProgressiveNet
        The Progressive Net model for Task-IL
    device : torch.device
        Device to run the model on
    tasks : List[torch.utils.data.TensorDataset]
        List of tasks as TensorDatasets
    replay_buffer : Optional[ReplayBuffer_TaskIL]
        Replay buffer for Task-IL (Default: None)
    replay_weight : float
        Weight for replay loss (Default: 1.0)
    is_ewc : bool
        Whether to use EWC regularization (Default: False)
    lambda_ewc : float
        EWC regularization strength (Default: 0.4)
    batch_size : int
        Batch size for training (Default: 64)
    epochs : int
        Number of epochs per task (Default: 10)
    gamma : float
        Learning rate decay factor (Default: 0.5)
    learning_rate : float
        Initial learning rate (Default: 0.001)
    loss_function : str
        Loss function to use (Default: "cross_entropy")
    step_size : int
        Step size for learning rate scheduler (Default: 20)
    weight_decay : float
        Weight decay for optimizer (Default: 0.00001)
    print_every : int
        Print progress every n epochs (Default: 1)

    Returns
    -------
    Tuple[ProgressiveNet, AccuracyMatrix]
        Trained model and accuracy matrix across tasks

    Usage Example
    --------------
    model, accuracy_matrix = full_training_prog_net_task_il(model, device, tasks)
    """
    num_tasks = len(tasks)
    accuracy_matrix = AccuracyMatrix(num_tasks=num_tasks)
    ewc = None
    
    for t, task in enumerate(tasks):
        print(f"Training on Task {t} with {len(task)} samples")
        model.add_column()

        # Train on current task with replay buffer and ewc if provided
        model = train_prog_net_on_task_task_il(
            model=model,
            device=device,
            train_dataset=task,
            task_id=t, 
            batch_size=batch_size,
            epochs=epochs,
            gamma=gamma,
            learning_rate=learning_rate,
            loss_function=loss_function,
            step_size=step_size,
            weight_decay=weight_decay,
            replay_buffer=replay_buffer,
            replay_weight=replay_weight,
            ewc=ewc,
            print_every=print_every
        )

        # Evaluate on all seen tasks
        for i in range(t + 1):
            acc = evaluate_task_task_il(
                model=model, 
                dataset=tasks[i],
                task_id=i,
                device=device,
                batch_size=batch_size
            )
            accuracy_matrix.update(task_i=i, task_t=t, accuracy=acc)
            print(f"  Eval on Task {i}: Accuracy = {acc*100:.2f}%")

        # Update replay buffer after task
        if replay_buffer is not None:
            replay_buffer.add(task_id=t, X=task.tensors[0], y=task.tensors[1])

        # Update EWC after task
        if is_ewc:
            task_loader = make_dataloaders(dataset=task, batch_size=batch_size, shuffle=True)
            ewc = EWC_TaskIL(model=model, dataloader=task_loader, task_id=t, device=device, lambda_ewc=lambda_ewc)

    return model, accuracy_matrix


# =======================================
# Full Training Pipeline on DEN and for Class-IL
# =======================================
def full_training_den(
    model: DEN, 
    device: torch.device,
    tasks: List[torch.utils.data.TensorDataset],
    ouptut_dim: int,
    replay_buffer: Optional[ReplayBuffer_ClassIL] = None,
    replay_weight: float = 1.0,
    is_ewc: bool = False,
    lambda_ewc: float = 0.4,
    batch_size: int = 64,
    epochs: int = 10,
    gamma: float = 0.5,
    learning_rate: float = 0.001,
    step_size: int = 20,
    weight_decay: float = 0.00001,
    lambda_sparse: float = 0.001,
    max_grad_norm: float = 1.0,
    grad_threshold: float = 0.1,
    percentile: float = 20.0,
    print_every: int = 1
) -> Tuple[DEN, AccuracyMatrix]:
    """
    Full training pipeline for Class-IL scenario using Dynamic Expandable Network (DEN) with optional Replay and EWC

    Parameters
    ----------
    model : DEN
        The DEN model for Class-IL
    device : torch.device
        Device to run the model on
    tasks : List[torch.utils.data.TensorDataset]
        List of tasks as TensorDatasets
    ouptut_dim : int
        Total number of classes across all tasks
    replay_buffer : Optional[ReplayBuffer_ClassIL]
        Replay buffer for Class-IL (Default: None)
    replay_weight : float
        Weight for replay loss (Default: 1.0)
    is_ewc : bool
        Whether to use EWC regularization (Default: False)
    lambda_ewc : float
        EWC regularization strength (Default: 0.4)
    batch_size : int
        Batch size for training (Default: 64)
    epochs : int
        Number of epochs per task (Default: 10)
    gamma : float
        Learning rate decay factor (Default: 0.5)
    learning_rate : float
        Initial learning rate (Default: 0.001)
    step_size : int
        Step size for learning rate scheduler (Default: 20)
    weight_decay : float
        Weight decay for optimizer (Default: 0.00001)
    lambda_sparse : float
        Sparsity regularization strength (Default: 0.001)
    max_grad_norm : float
        Maximum gradient norm for clipping (Default: 1.0)
    grad_threshold : float
        Gradient threshold for expansion (Default: 0.1)
    percentile : float
        Percentile of neurons to freeze based on importance after expansion (Default: 20.0)
    print_every : int
        Print progress every n epochs (Default: 1)

    Returns
    -------
    Tuple[DEN, AccuracyMatrix]
        Trained model and accuracy matrix across tasks

    Usage Example
    --------------
    model, accuracy_matrix = full_training_den(model, device, tasks)
    """
    num_tasks = len(tasks)
    num_classes_per_task = ouptut_dim // num_tasks
    accuracy_matrix = AccuracyMatrix(num_tasks=num_tasks)
    ewc = None
    
    for t, task in enumerate(tasks):
        print(f"Training on Task {t} with {len(task)} samples")

        # Train on current task with replay buffer and ewc if provided
        model = train_den_on_task(
            model=model,
            device=device,
            train_dataset=task,
            task_classes=list(range(t * num_classes_per_task, (t + 1) * num_classes_per_task)), # assume 2 classes per task, sorted
            batch_size=batch_size,
            epochs=epochs,
            gamma=gamma,
            learning_rate=learning_rate,
            step_size=step_size,
            weight_decay=weight_decay,
            lambda_sparse=lambda_sparse,
            max_grad_norm=max_grad_norm,
            replay_buffer=replay_buffer,
            replay_weight=replay_weight,
            ewc=ewc,
            print_every=print_every
        )

        # Evaluate on all seen tasks
        for i in range(t + 1):
            acc = evaluate_task_den(
                model=model, 
                dataset=tasks[i],
                seen_classes=list(range((i + 1) * 2)),  # Assuming 2 classes per task
                device=device,
                batch_size=batch_size
            )
            accuracy_matrix.update(task_i=i, task_t=t, accuracy=acc)
            print(f"  Eval on Task {i}: Accuracy = {acc*100:.2f}%")

        # Update replay buffer after task
        if replay_buffer is not None:
            replay_buffer.add(X=task.tensors[0], y=task.tensors[1])

        # Update EWC after task
        if is_ewc:
            task_loader = make_dataloaders(dataset=task, batch_size=batch_size, shuffle=True)
            ewc = EWC(model=model, dataloader=task_loader, device=device, lambda_ewc=lambda_ewc)

        # See if expansion is needed after task
        if needs_expansion(model, grad_threshold=grad_threshold):
            print("Expanding DEN model...")
            expand_den(model, device=device)
        
        # Freeze low-importance neurons after task
        freeze_neurons(model, percentile=percentile)

    return model, accuracy_matrix


# =======================================
# Final calibration phase for DEN after all tasks
# =======================================
def calibrate_den(
    den_model: DEN,
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0.00001
):
    """
    Final calibration phase for DEN after all tasks have been learned.
    Freezes feature extractor and retrains only the classifier on all seen data.

    Parameters
    ----------
    den_model : DEN
        The trained DEN model
    device : torch.device
        Device to run the model on
    X_train : np.ndarray
        Training data features (all seen data)
    y_train : np.ndarray
        Training data labels (all seen data)
    batch_size : int
        Batch size for training (Default: 64)
    epochs : int
        Number of epochs for calibration (Default: 10)
    learning_rate : float
        Learning rate for optimizer (Default: 0.001)
    weight_decay : float
        Weight decay for optimizer (Default: 0.00001)

    Returns
    -------
    None

    Usage Example
    --------------
    calibrate_den(den_model, device, X_train, y_train)
    """
    den_model.to(device)
    # freeze feature extractor
    for layer in den_model.layers:
        layer.freeze_existing()

    # retrain ONLY classifier on all seen data
    den_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(den_model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    full_train_loader = make_dataloaders(df_to_tensor_dataset(X_train, y_train), batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in full_train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = den_model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        print(f"[Calibration] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(full_train_loader.dataset):.4f}")

    return den_model


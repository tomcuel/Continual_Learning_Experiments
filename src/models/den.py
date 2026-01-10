# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# =======================================
# DEN Layer for Dynamic Expansion Network
# =======================================
class DENLayer(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int
    ):
        """
        DEN Layer with neuron freezing and expansion capabilities
        
        Parameters
        ----------
        in_dim : int
            Input feature dimension
        out_dim : int
            Output feature dimension (number of neurons)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.full((out_dim,), 0.1))

        # frozen mask: False = neuron is trainable
        self.frozen_mask = torch.zeros(out_dim, dtype=torch.bool)

    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with frozen neuron handling

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_dim)

        Usage Example
        --------------
        out = den_layer(x)
        """
        # compute linear transformation
        out = F.linear(x, self.weight, self.bias)

        # detach outputs of frozen neurons to prevent gradient flow
        if self.frozen_mask.any():
            out = out.clone()  # ensure we don't modify in-place
            out[:, self.frozen_mask] = out[:, self.frozen_mask].detach()

        return out

    def freeze_existing(self):
        """
        Freeze all existing neurons in the layer

        Returns
        -------
        None

        Usage Example
        --------------
        den_layer.freeze_existing()
        """
        self.frozen_mask[:] = True

    def expand(self,
        k: int
    ):
        """
        Expand the layer by adding k new neurons

        Parameters
        ----------
        k : int
            Number of neurons to add

        Returns
        -------
        None

        Usage Example
        --------------
        den_layer.expand(k)
        """
        # add new neurons
        new_w = nn.Parameter(torch.empty(k, self.in_dim))
        nn.init.kaiming_normal_(new_w, nonlinearity='relu')
        new_b = nn.Parameter(torch.full((k,), 0.1))

        self.weight = nn.Parameter(torch.cat([self.weight, new_w], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias, new_b], dim=0))
        self.frozen_mask = torch.cat([self.frozen_mask, torch.zeros(k, dtype=torch.bool)])
        self.out_dim += k


# =======================================
# Dynamic Expansion Network (DEN)
# =======================================
class DEN(nn.Module):
    def __init__(self, 
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ):
        """
        Dynamic Expansion Network (DEN) model

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dims : List[int]
            List of hidden layer dimensions
        output_dim : int
            Output feature dimension (number of classes)

        Returns
        -------
        None

        Usage Example
        --------------
        model = DEN(input_dim=784, hidden_dims=[256, 256], output_dim=10)
        """
        super().__init__()
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList()
        prev = input_dim

        for h in hidden_dims:
            self.layers.append(DENLayer(prev, h))
            prev = h

        self.classifier = nn.Linear(prev, output_dim)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the DEN model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim)

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (N, output_dim)

        Usage Example
        --------------
        logits = model(x)
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.classifier(x)
    

# =======================================
# DEN expansion logic: expand only if new task cannot be learned well
# =======================================
def needs_expansion(
    model: DEN,
    grad_threshold: float = 1e-3
) -> bool:
    """
    Determine if the model needs to expand based on gradients of frozen neurons

    Parameters
    ----------
    model : DEN
        The DEN model
    grad_threshold : float
        Gradient norm threshold to decide expansion

    Returns
    -------
    bool
        True if expansion is needed, False otherwise

    Usage Example
    --------------
    if needs_expansion(model):
        expand_den(model, k)
    """
    conflict = 0.0
    count = 0

    for layer in model.layers:
        if layer.frozen_mask.any() and layer.weight.grad is not None:
            frozen_grads = layer.weight.grad[layer.frozen_mask]
            conflict += frozen_grads.norm().item()
            count += frozen_grads.numel()

    return (conflict / (count + 1e-8)) > grad_threshold


def expand_den(
    model: DEN,
    percentage: float = 20.0
):
    """
    Expand the DEN model by adding k neurons to each layer

    Parameters
    ----------
    model : DEN
        The DEN model
    percentage : float
        Percentage of current neurons to add as new neurons in each layer

    Returns
    -------
    None

    Usage Example
    --------------
    expand_den(model, percentage=20.0)
    """
    for i, layer in enumerate(model.layers):
        k = int(layer.out_dim * percentage)
        layer.expand(k)

        # expand next layer input
        if i + 1 < len(model.layers):
            next_layer = model.layers[i + 1]

            old_w = next_layer.weight.detach()
            # use proper Kaiming init for new columns
            new_cols = nn.init.kaiming_normal_(torch.empty(next_layer.out_dim, k), nonlinearity='relu')

            next_layer.weight = nn.Parameter(torch.cat([old_w, new_cols], dim=1))
            next_layer.in_dim += k

    # expand classifier input
    old_w = model.classifier.weight.detach()
    old_b = model.classifier.bias.detach()

    out_dim, old_in = old_w.shape
    new_w = nn.init.kaiming_normal_(torch.empty(out_dim, k), nonlinearity='linear')

    model.classifier = nn.Linear(old_in + k, out_dim)
    model.classifier.weight.data = torch.cat([old_w, new_w], dim=1)
    model.classifier.bias.data.copy_(old_b)


# =======================================
# Loss function for task specific classes
# =======================================
def class_il_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_classes: List[int]
) -> torch.Tensor:
    """
    Compute loss only over the classes relevant to the current task

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (N, total_classes)
    targets : torch.Tensor
        True labels of shape (N,)
    task_classes : List[int]
        List of class indices relevant to the current task

    Returns
    -------
    torch.Tensor
        Computed cross-entropy loss

    Usage Example
    --------------
    loss = class_il_loss(logits, targets, task_classes)
    """
    # extract only task logits
    task_logits = logits[:, task_classes]

    # remap labels: {2,3} → {0,1}
    label_map = {c: i for i, c in enumerate(task_classes)}
    task_targets = torch.tensor(
        [label_map[int(y)] for y in targets],
        device=targets.device
    )

    return F.cross_entropy(task_logits, task_targets)


# =======================================
# Freeze LOW-importance neurons
# =======================================
def freeze_neurons(
    model: DEN,
    percentile: float = 20.0
):
    """
    Freeze bottom X% of neurons in each layer based on gradient importance (only call after a task has converged)

    Parameters
    ----------
    model : DEN
        The DEN model
    percentile : float
        Percentile of neurons to freeze based on importance

    Returns
    -------
    None

    Usage Example
    --------------
    freeze_neurons(model, percentile=20.0)
    """
    for layer in model.layers:
        if layer.weight.grad is None:
            continue

        # gradient norm per neuron
        grad_norm = layer.weight.grad.norm(dim=1)
        importance = grad_norm / (grad_norm.max() + 1e-8)

        # freeze bottom X% neurons
        cutoff = torch.quantile(importance, percentile / 100)
        layer.frozen_mask |= importance < cutoff


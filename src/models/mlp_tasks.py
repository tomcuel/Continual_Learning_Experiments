# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


from src.data.torch_utilities import _ACTIVATION_MAP


# =======================================
# Model Definition for Task-IL (Multi-head)
# =======================================
class DeepNN_TaskIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        num_tasks: int = 5,
        output_dim_per_task: int = 2,   # binary per task
        activations: Optional[List[str]] = None,
        dropout_rates: Optional[List[float]] = None,
        use_batchnorm: bool = True
    ):
        """
        Deep Neural Network for Task-IL with task-specific heads

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_layers : List[int]
            List of hidden layer sizes
        num_tasks : int
            Number of tasks
        output_dim_per_task : int
            Output dimension per task
        activations : Optional[List[str]]
            List of activation functions for each hidden layer
        dropout_rates : Optional[List[float]]
            List of dropout rates for each hidden layer
        use_batchnorm : bool
            Whether to use batch normalization after each hidden layer
        """
        super().__init__()
        layers = []
        prev = input_dim
        n_hidden = len(hidden_layers)

        if activations is None:
            activations = ["relu"] * n_hidden
        if dropout_rates is None:
            dropout_rates = [0.0] * n_hidden
        assert len(activations) == n_hidden
        assert len(dropout_rates) == n_hidden

        # Shared Backbone
        for idx, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            act_name = activations[idx]
            act_ctor = _ACTIVATION_MAP.get(act_name)
            if act_ctor is None:
                raise ValueError(f"Unsupported activation: {act_name}")
            if act_ctor is not None:
                layers.append(act_ctor())
            if dropout_rates[idx] and dropout_rates[idx] > 0:
                layers.append(nn.Dropout(dropout_rates[idx]))
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Task-specific Heads
        self.heads = nn.ModuleList([nn.Linear(prev, output_dim_per_task) for _ in range(num_tasks)])

        self._init_weights(activations)

    def _init_weights(self, 
        activations: List[str]
    ):
        """
        Initialize weights of the network based on activation functions

        Parameters
        ----------
        activations : List[str]
            List of activation functions for each hidden layer

        Returns
        -------
        None

        Usage Example
        --------------
        model._init_weights(activations)
        """
        idx_hidden = 0
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                # choose activation of the incoming layer index
                act = activations[idx_hidden] if idx_hidden < len(activations) else None
                if act in ("relu", "leaky_relu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu") # He init
                else:
                    nn.init.xavier_normal_(m.weight) # Xavier init
                nn.init.zeros_(m.bias)
                idx_hidden += 1

        # Init heads
        for head in self.heads:
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, 
        x: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        """
        Forward pass through the DeepNN_TaskIL model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim)
        task_id : int
            Task identifier to select the appropriate head

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (N, output_dim_per_task)

        Usage Example
        --------------
        logits = model(x, task_id)
        """
        features = self.backbone(x)
        return self.heads[task_id](features)
    

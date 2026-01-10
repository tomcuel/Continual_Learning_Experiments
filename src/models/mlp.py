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
# Model Definition
# =======================================
class DeepNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activations: Optional[List[str]] = None,
        dropout_rates: Optional[List[float]] = None,
        use_batchnorm: bool = True
    ):
        """
        Deep Neural Network Model

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_layers : List[int]
            List of hidden layer sizes
        output_dim : int
            Output feature dimension
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

        for idx, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            act_name = activations[idx]
            act_ctor = _ACTIVATION_MAP.get(act_name)
            if act_ctor is None and act_name != "softmax_logit":
                raise ValueError(f"Unsupported activation: {act_name}")
            if act_ctor is not None:
                layers.append(act_ctor())
            if dropout_rates[idx] and dropout_rates[idx] > 0:
                layers.append(nn.Dropout(dropout_rates[idx]))
            prev = h
        
        # Output layer (no activation here if using CrossEntropy)
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

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
        for m in self.net:
            if isinstance(m, nn.Linear):
                # choose activation of the incoming layer index
                act = activations[idx_hidden] if idx_hidden < len(activations) else None
                if act in ("relu", "leaky_relu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu") # He init
                else:
                    nn.init.xavier_normal_(m.weight) # Xavier init
                nn.init.zeros_(m.bias)
                idx_hidden += 1

    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the DeepNN model

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
        return self.net(x)
    
    
# ===============================
# Library Imports
# ===============================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# =======================================
# Progressive Column
# =======================================
class ProgColumn(nn.Module):
    def __init__(self, 
        input_dim: int,
        hidden_dims: List[int]
    ):
        """
        Progressive Network Column
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dims : List[int]
            List of hidden layer sizes
        """
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev, h))
            prev = h

    def forward(self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the column

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim)

        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            Output tensor of shape (N, last_hidden_dim) and list of activations per layer

        Usage Example
        --------------
        out, activations = prog_column(x)
        """
        activations = []
        for layer in self.layers:
            x = torch.relu(layer(x))
            activations.append(x)
        return x, activations
    

# =======================================
# Progressive Network (Class-IL compatible)
# =======================================
class ProgressiveNet(nn.Module):
    def __init__(self, 
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ):
        """
        Progressive Network for Continual Learning

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dims : List[int]
            List of hidden layer sizes for each column
        output_dim : int
            Output feature dimension (number of classes)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.columns = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.lateral = nn.ModuleList()

    def add_column(self):
        """
        Add a new column to the Progressive Network

        Returns
        -------
        None

        Usage Example
        --------------
        model.add_column()
        """
        col = ProgColumn(self.input_dim, self.hidden_dims)
        self.columns.append(col)

        head = nn.Linear(self.hidden_dims[-1], self.output_dim)
        self.heads.append(head)

        # Lateral connections
        if len(self.columns) > 1:
            lateral_layers = nn.ModuleList()
            for h in self.hidden_dims:
                lateral_layers.append(nn.Linear(h, h, bias=False))
            self.lateral.append(lateral_layers)

    def forward(self, 
        x: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        """
        Forward pass through the Progressive Network

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim)
        task_id : int
            Task/column ID to forward through

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (N, output_dim)

        Usage Example
        --------------
        logits = model(x, task_id)
        """
        activations_per_column = []

        # Forward through all columns up to task_id
        for col in self.columns[:task_id + 1]:
            _, acts = col(x)
            activations_per_column.append(acts)

        # Forward through current column again to get final hidden
        h, acts_current = self.columns[task_id](x)

        # Add lateral connections
        if task_id > 0:
            for l, lat in enumerate(self.lateral[task_id - 1]):
                for k in range(task_id):
                    h = h + lat(activations_per_column[k][l])

        return self.heads[task_id](h)


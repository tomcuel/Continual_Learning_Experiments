# =======================================
# Library Imports
# =======================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple


# =======================================
# EWC Implementation
# =======================================
class EWC:
    def __init__(self, 
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        lambda_ewc: float = 0.4
    ):
        """
        Elastic Weight Consolidation (EWC) implementation
        
        Parameters
        ----------
        model : nn.Module
            The neural network model
        dataloader : DataLoader
            DataLoader for the previous task
        device : torch.device
            Device to perform computations on
        lambda_ewc : float
            Regularization strength for EWC penalty
        """
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, 
        dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Fisher Information Matrix
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for the previous task
        
        Returns
        -------
        fisher : Dict[str, torch.Tensor]
            Fisher Information for each parameter
        
        Usage Example
        --------------
        fisher = self._compute_fisher(dataloader)
        """
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():  # explicit, controlled
                output = self.model(X)
                loss = criterion(output, y)
                loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= len(dataloader)

        return fisher

    def penalty(self, 
        model: nn.Module
    ) -> torch.Tensor:
        """
        Compute the EWC penalty

        Parameters
        ----------
        model : nn.Module
            The current neural network model

        Returns
        -------
        loss : torch.Tensor
            EWC penalty loss

        Usage Example   
        --------------
        loss = self.penalty(model)
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lambda_ewc * loss
    

# =======================================
# EWC Implementation for Task-IL
# =======================================
class EWC_TaskIL:
    def __init__(self, 
        model: nn.Module,
        dataloader: DataLoader,
        task_id: int,
        device: torch, 
        lambda_ewc: float = 0.4
    ):
        """
        Elastic Weight Consolidation (EWC) implementation for Task-IL

        Parameters
        ----------
        model : nn.Module
            The neural network model
        dataloader : DataLoader
            DataLoader for the previous task
        task_id : int
            The task identifier
        device : torch.device
            Device to perform computations on
        lambda_ewc : int
            Regularization strength for EWC penalty
        """
        self.model = model
        self.task_id = task_id
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, 
        dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Fisher Information Matrix for Task-IL

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for the previous task

        Returns
        -------
        fisher : Dict[str, torch.Tensor]
            Fisher Information for each parameter

        Usage Example
        --------------
        fisher = self._compute_fisher(dataloader)
        """
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.model.zero_grad()
            output = self.model(X, self.task_id)
            loss = criterion(output, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= len(dataloader)

        return fisher

    def penalty(self, 
        model: nn.Module
    ) -> torch.Tensor:
        """
        Compute the EWC penalty for Task-IL

        Parameters
        ----------
        model : nn.Module
            The current neural network model

        Returns
        -------
        loss : torch.Tensor
            EWC penalty loss

        Usage Example
        --------------
        loss = self.penalty(model)
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lambda_ewc * loss


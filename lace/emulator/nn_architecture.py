import numpy as np
import torch
from torch import nn


class MDNemulator_polyfit(torch.nn.Module):
    """
    A neural network model for emulating power spectrum P1D using polynomial fitting.

    Args:
        nhidden (int): Number of hidden layers.
        ndeg (int): Degree of polynomial fit.
        max_neurons (int, optional): Maximum number of neurons in any layer. Defaults to 100.
        ninput (int, optional): Number of input features. Defaults to 6.
    """

    def __init__(
        self, nhidden, ndeg, max_neurons=100, ninput=6, pred_error=False
    ):
        super().__init__()
        self.pred_error = pred_error
        self.inputlay = torch.nn.Sequential(
            nn.Linear(ninput, max_neurons), nn.LeakyReLU(0.5)
        )

        modules = []
        for k in range(nhidden - 1):
            modules.append(nn.Linear(max_neurons, max_neurons))
            modules.append(nn.LeakyReLU(0.5))
        self.hiddenlay = nn.Sequential(*modules)

        self.means = torch.nn.Sequential(
            nn.Linear(max_neurons, ndeg),
        )

        if self.pred_error:
            self.stds = torch.nn.Sequential(
                nn.Linear(max_neurons, 50),
                nn.LeakyReLU(0.5),
                nn.Linear(50, ndeg),
            )

    def forward(self, inp):
        """
        Forward pass of the neural network.

        Args:
            inp (torch.Tensor): Input tensor containing features.

        Returns:
            tuple:
                - torch.Tensor: Emulated P1D values.
                - torch.Tensor: Logarithm of the standard deviation of P1D values.
        """
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        p1d = self.means(x)

        if self.pred_error:
            logerrp1d = self.stds(x)
            return p1d, logerrp1d
        else:
            return p1d

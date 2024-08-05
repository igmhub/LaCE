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
    def __init__(self, nhidden, ndeg, max_neurons=100, ninput=6):
        super().__init__()
        self.inputlay = torch.nn.Sequential(
            nn.Linear(ninput, 10), nn.LeakyReLU(0.5)
        )

        params = np.linspace(10, max_neurons, nhidden)
        modules = []
        for k in range(nhidden - 1):
            modules.append(nn.Linear(int(params[k]), int(params[k + 1])))
            modules.append(nn.LeakyReLU(0.5))
        self.hiddenlay = nn.Sequential(*modules)

        self.means = torch.nn.Sequential(
            nn.Linear(max_neurons, 50), nn.LeakyReLU(0.5), nn.Linear(50, ndeg + 1)
        )
        self.stds = torch.nn.Sequential(
            nn.Linear(max_neurons, 50), nn.LeakyReLU(0.5), nn.Linear(50, ndeg + 1)
        )

    def forward(self, inp):
        """
        Forward pass of the neural network.

        Args:
            inp (torch.Tensor): Input tensor containing features.

        Returns:
            tuple: 
                - p1d (torch.Tensor): Emulated P1D values.
                - logerrp1d (torch.Tensor): Logarithm of the standard deviation of P1D values.
        """
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        p1d = self.means(x)
        logerrp1d = self.stds(x)

        return p1d, logerrp1d

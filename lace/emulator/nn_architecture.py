import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.optim import lr_scheduler


class MDNemulator_polyfit(torch.nn.Module):
    def __init__(self, nhidden, ndeg, ninput=6):
        super().__init__()
        self.inputlay = torch.nn.Sequential(
            nn.Linear(ninput, 10), nn.LeakyReLU(0.5)
        )

        params = np.linspace(10, 100, nhidden)
        modules = []
        for k in range(nhidden - 1):
            modules.append(nn.Linear(int(params[k]), int(params[k + 1])))
            modules.append(nn.LeakyReLU(0.5))
        self.hiddenlay = nn.Sequential(*modules)

        self.means = torch.nn.Sequential(
            nn.Linear(100, 50), nn.LeakyReLU(0.5), nn.Linear(50, ndeg + 1)
        )
        self.stds = torch.nn.Sequential(
            nn.Linear(100, 50), nn.LeakyReLU(0.5), nn.Linear(50, ndeg + 1)
        )

    def forward(self, inp):
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        p1d = self.means(x)
        logerrp1d = self.stds(x)

        return p1d, logerrp1d

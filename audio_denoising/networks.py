import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter


class VariationalParams(Module):
    def __init__(self, N, D, H) -> None:
        super().__init__()
        self.N = N  # number of data points
        self.D = D  # observed space dimensionality
        self.H = H  # latent space dimensionality


    def forward(self, X, indexes):
        raise NotImplementedError()
    

class ResBlock(nn.Module):
    def __init__(self, auto_block=None):
        nn.Module.__init__(self)
        self.auto_block = auto_block

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.auto_block(x)
    

class ResNetSequenceVariationalParams(VariationalParams):
    def __init__(self, N, D, H):
        super().__init__(N, D, H)
        dim = [D, 5*D, 5*D]
        
        self.nn_common = nn.Sequential(
            nn.Linear(D, dim[1]),
            nn.ReLU(),
            ResBlock(nn.Sequential(
                nn.Linear(dim[1], dim[2]),
                nn.ReLU(),
                nn.Linear(dim[2], dim[1]),
                nn.ReLU(),
            )),
            nn.BatchNorm1d(dim[1]),
        )

        self.nn_mean = nn.Sequential(
            ResBlock(nn.Sequential(
                nn.Linear(dim[1], dim[2]),
                nn.ReLU(),
                nn.Linear(dim[2], dim[1]),
                nn.ReLU(),
            )),
            nn.BatchNorm1d(dim[1]),
            nn.Linear(dim[1], H),
        )

        self.nn_cond = nn.Sequential(
            ResBlock(nn.Sequential(
                nn.Linear(dim[1], dim[2]),
                nn.ReLU(),
                nn.Linear(dim[2], dim[1]),
                nn.ReLU(),
            )),
            nn.BatchNorm1d(dim[1]),
            nn.Linear(dim[1], int((H*H-H)/2)),
        )

    def forward(self, X, indexes):
        common = self.nn_common(X)
        M_params = self.nn_mean(common)
        C_params = self.nn_cond(common)
        return M_params, C_params
    

class ResNetMarginalVariationalParams(VariationalParams):
    def __init__(self, N, D, H):
        super().__init__(N, D, H)
        dim = [D, 5*D, 5*D]
        
        self.nn_common = nn.Sequential(
            nn.Linear(D, dim[1]),
            nn.ReLU(),
            ResBlock(nn.Sequential(
                nn.Linear(dim[1], dim[2]),
                nn.ReLU(),
                nn.Linear(dim[2], dim[1]),
                nn.ReLU(),
            )),
            nn.BatchNorm1d(dim[1]),
        )

        self.nn_mean = nn.Sequential(
            ResBlock(nn.Sequential(
                nn.Linear(dim[1], dim[2]),
                nn.ReLU(),
                nn.Linear(dim[2], dim[1]),
                nn.ReLU(),
            )),
            nn.BatchNorm1d(dim[1]),
            nn.Linear(dim[1], H),
        )


    def forward(self, X, indexes):
        common = self.nn_common(X)
        M_params = self.nn_mean(common)
        return M_params
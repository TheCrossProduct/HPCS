import torch
import torch.nn as nn
from hpcs.nn import MLP
from hpcs.utils.poincare import expmap_1

class ExpMap(nn.Module):
    def __init__(self):
        super(ExpMap, self).__init__()
    def forward(self, x):
        return expmap_1(x, torch.zeros_like(x))

class HypNN(nn.Module):
    def __init__(self):
        pass

class MLPExpMap(nn.Module):
    def __init__(self, input_feat: int, out_feat: int, bias: bool = False, negative_slope: float = 0.2, dropout: float = 0.0):
        super(MLPExpMap, self).__init__()
        self.mlp = MLP([input_feat, out_feat], bias=bias, negative_slope=negative_slope, dropout=dropout)

    def forward(self, x):
        x = self.mlp(x)
        return expmap_1(x, torch.zeros_like(x))
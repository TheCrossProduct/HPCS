import torch
import torch.nn as nn

class ExpMap(nn.Module):
    def __init__(self, in_channels, out_features, scale):
        super(ExpMap, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.scale = nn.Parameter(torch.Tensor([scale]))

    def forward(self, x):
        raise NotImplemented
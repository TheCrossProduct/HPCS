import torch
import math

from torch.nn import functional as F, Linear, Sequential as Seq, BatchNorm1d, Dropout


class EulerFeatExtract(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_features: int, negative_slope: float = 0.2, bias: bool = True,
                 dropout: float = 0.0, init_gamma: float = math.pi / 2):
        super(EulerFeatExtract, self).__init__()

        self.gamma = torch.nn.Parameter(torch.Tensor([init_gamma]), requires_grad=True)

        self.mlp1 = MLP([in_channels, hidden_features, hidden_features], negative_slope=negative_slope,
                        dropout=dropout, bias=bias)

        self.mlp2 = MLP([in_channels, hidden_features, hidden_features], negative_slope=negative_slope, dropout=dropout,
                        bias=bias)
        self.linear = Seq(
            Linear(in_features=hidden_features, out_features=1, bias=True),
            BatchNorm1d(1)
        )

    def forward(self, x):
        x = self.mlp1(x)
        # x2 = self.mlp2(x)
        # return torch.cat([x1, x2], dim=1)
        x = self.linear(x)
        # x = (x - x.min()) / (x.max() - x.min())
        return torch.cat([torch.cos(self.gamma * x), torch.sin(self.gamma * x)], dim=1)
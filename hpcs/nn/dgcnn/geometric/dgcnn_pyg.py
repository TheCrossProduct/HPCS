import torch

from torch_geometric.nn import DynamicEdgeConv

from hpcs.nn.dgcnn.utils.transform_net import Transform_Net
from torch_geometric.nn import MLP

class DGCNN(torch.nn.Module):
    def __init__(self, in_channels: int, out_features: int, hidden_features: int, k: int, transformer: bool = False,
                 negative_slope: float = 0.2, dropout=0.0, cosine=False):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.k = k
        self.negative_slope = negative_slope
        self.cosine = cosine
        self.transformer = transformer

        if self.transformer:
            self.tnet = Transform_Net()

        self.conv1 = DynamicEdgeConv(
            nn=MLP([2 * in_channels, hidden_features, hidden_features, hidden_features], dropout=dropout, negative_slope=self.negative_slope),
            k=self.k,
            cosine=False,
        )
        self.conv2 = DynamicEdgeConv(
            nn=MLP([2 * hidden_features, hidden_features, hidden_features, hidden_features], dropout=dropout, negative_slope=self.negative_slope),
            k=self.k,
            cosine=False,
        )
        self.conv3 = DynamicEdgeConv(
            nn=MLP([2 * hidden_features, hidden_features, hidden_features, self.out_features], dropout=dropout, negative_slope=self.negative_slope),
            k=self.k,
            cosine=self.cosine,
        )


    def forward(self, pos, batch=None):
        # x = torch.cat([x, pos], dim=-1)
        x = pos
        x = self.conv1(x, batch=batch)
        x = self.conv2(x, batch=batch)
        x = self.conv3(x, batch=batch)

        return x
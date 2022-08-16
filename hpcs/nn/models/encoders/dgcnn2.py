import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv


class DGCNN2(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, dropout: int, aggr='max'):
        super(DGCNN2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.dropout = dropout

        self.conv1 = DynamicEdgeConv(MLP([2 * self.in_channels, 64, 64], dropout=self.dropout), self.k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64], dropout=self.dropout), self.k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64], dropout=self.dropout), self.k, aggr)

        self.mlp = MLP([3 * 64, 1024, 256, 128, self.out_channels], dropout=0.5, norm=None)

    def forward(self, x, pos, batch):
        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch=batch)
        x2 = self.conv2(x1, batch=batch)
        x3 = self.conv3(x2, batch=batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return F.log_softmax(out, dim=1)

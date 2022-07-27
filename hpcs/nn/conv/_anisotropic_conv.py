import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.nn.inits import ones
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter

from hpcs.nn.models.networks._mlp import MLP
from ..__init__ import init_weights

class BilateralConv(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int,
                 dropout: float = 0.0,
                 negative_slope: float = 0.2,
                 add_self_loops: bool = True,
                 bias: bool = True,
                 **kwargs):
        super(BilateralConv, self).__init__(aggr='add', **kwargs)
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.bias = bias

        self.nn = MLP([in_channels, out_channels], negative_slope=self.negative_slope, bias=self.bias)
        self.conv = MLP([2 * out_channels, out_channels], negative_slope=self.negative_slope, bias=self.bias)

        self.r = Parameter(Tensor(1), requires_grad=True)
        self.s = Parameter(Tensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init_weights(self.nn, xavier_normal_)
        init_weights(self.conv, xavier_normal_)
        ones(self.r)
        ones(self.s)

    def forward(self, x, pos, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        x1 = torch.cat([self.nn(x), pos], dim=1)

        out = self.propagate(edge_index=edge_index, x=x1, size=None)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        h_i = x_i[:, :self.out_channels]
        pos_i = x_i[:, self.out_channels:]
        h_j = x_j[:, :self.out_channels]
        pos_j = x_i[:, self.out_channels:]

        bil_filtering = torch.exp_(-(torch.norm(h_i - h_j, dim=1) / (self.r ** 2 + 1e-16)) - (torch.norm(pos_i - pos_j) / (self.s ** 2 + 1e-16)))

        N = maybe_num_nodes(index, num_nodes=size_i)
        bil_sum = scatter(bil_filtering, index, dim=0, dim_size=N, reduce='sum')[index]
        bil_filtering = bil_filtering / (bil_sum + 1e-16)

        return self.conv(torch.cat([h_i - h_j, h_i], dim=1)) * bil_filtering.view(-1, 1)

    def __repr__(self):
        return '{}({}, {}, k={})'.format(self.__class__.__name__,
                                         self.in_channels,
                                         self.out_channels, self.k)

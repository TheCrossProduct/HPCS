from typing import Union, Optional
from torch_geometric.typing import OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import relu
from torch_scatter import scatter, segment_csr
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.nn.inits import normal, zeros, reset

from hpcs.nn.models._mlp import MLP


def dot_operation(x, weights):
    """
    In the (max, + ) algebra dot product is replaced by sum.
    This function implements the dot product between an input tensor x of shape (N, d)
    and a weight tensor of shape (d, h)
    """
    return torch.stack([torch.max(xel - relu(weights.T), dim=1)[0] for xel in x])


@torch.jit.script
def morpho_conv(x: Tensor, weight):
    n, k, in_channels = x.shape
    out_channels = weight.shape[-1]
    x_out = torch.zeros(n, k, out_channels, device=x.device)
    view_size = x.size()[1:] + torch.Size([1])
    expand_size = x.size()[1:] + torch.Size([out_channels])  # [N, k, in_channels, out_channels]

    for i in range(n):
        x_out[i] = torch.max(x[i].view(view_size).expand(expand_size) - relu(weight), dim=1)[0]

    # return torch.cat([torch.max(xn.view(view_size).expand(expand_size) - relu(weight), dim=1)[0] for xn in x])
    return x_out.reshape(-1, out_channels)


class DilateEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, nb_filters: int, k: int, flat_kernel: bool = False,
                 bias: bool = False, add_self_loops: bool = True):

        super(DilateEdgeConv, self).__init__(aggr='max')
        self.k = k
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.flat_kernels = flat_kernel
        self.add_self_loops = add_self_loops

        self.weights = Parameter(torch.Tensor(k, nb_filters), requires_grad=True)

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels * nb_filters), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        out = self.propagate(edge_index=edge_index[:, :x.shape[0]*self.k], x=x, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def message(self, x_j):
        # for the moment this implementation works only with two dimensional tensors
        x_ik = x_j.reshape(-1, self.k, x_j.shape[-1])
        x_out = x_ik.repeat_interleave(self.nb_filters, dim=-1)
        x_out = x_out - relu(self.weights.repeat_interleave(x_j.shape[-1], dim=-1))

        return x_out.reshape(-1, self.nb_filters * x_j.shape[-1])

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.nb_filters)


class DilateFlatEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, k: int, bias: bool = False, add_self_loops: bool = True):
        super(DilateFlatEdgeConv, self).__init__(aggr='max')
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.weights = Parameter(torch.Tensor(in_channels, out_channels), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        # op = torch.stack([torch.max(xel - relu(self.weights.T), dim=1)[0] for xel in x])
        x_out = x.unsqueeze(-1)
        op = x_out.expand(-1, -1, self.out_channels) - relu(self.weights)
        op = torch.max(op, dim=1)[0]

        out = self.propagate(edge_index=edge_index, x=op, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class DilateMaxPlus(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, k: int, flat_kernel: bool = False,
                 bias: bool = False, add_self_loops: bool = True):

        super(DilateMaxPlus, self).__init__(aggr='max')
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flat_kernels = flat_kernel
        self.add_self_loops = add_self_loops

        self.weights = Parameter(torch.Tensor(k, in_channels, out_channels), requires_grad=True)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        # strange behaviour of knn graph ... sometimes it returns empty edge index and some other more than K neighs
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        # print(x.shape, edge_index.shape)
        out = self.propagate(edge_index=edge_index[:, :(x.shape[0] * self.k)], x=x, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def message(self, x_j):
        # for the moment this implementation works only with two dimensional tensors
        x_ik = x_j.reshape(-1, self.k, self.in_channels)

        view_size = x_ik.size() + torch.Size([1])  # [N, k, in_channels, 1]
        expand_size = x_ik.size() + torch.Size([self.out_channels])  # [N, k, in_channels, out_channels]
        x_out = torch.max(x_ik.view(view_size).expand(expand_size) - relu(self.weights), dim=2)[0]
        return x_out.reshape(-1, self.out_channels)
        # return morpho_conv(x_ik, self.weights)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ErodeEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, nb_filters: int, k: int, flat_kernel: bool = False,
                 bias: bool = False, add_self_loops: bool = True):

        super(ErodeEdgeConv, self).__init__()
        self.aggr = "min"
        self.k = k
        self.in_channels = in_channels
        self.nb_filters = nb_filters
        self.flat_kernels = flat_kernel
        self.add_self_loops = add_self_loops

        self.weights = Parameter(torch.Tensor(k, nb_filters), requires_grad=True)

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels * nb_filters), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        out = self.propagate(edge_index=edge_index[:, :x.shape[0]*self.k], x=x, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def message(self, x_j):
        # for the moment this implementation works only with two dimensional tensors
        x_ik = x_j.reshape(-1, self.k, x_j.shape[-1])
        x_out = x_ik.repeat_interleave(self.nb_filters, dim=-1)
        x_out = x_out + relu(self.weights.repeat_interleave(x_j.shape[-1], dim=-1))

        return x_out.reshape(-1, self.nb_filters * x_j.shape[-1])

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.nb_filters)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """

        if ptr is not None:
            for _ in range(self.node_dim):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ErodeFlateEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, k: int, bias: bool = False, add_self_loops: bool = True):
        super(ErodeFlateEdgeConv, self).__init__()
        self.aggr = "min"
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.weights = Parameter(torch.Tensor(in_channels, out_channels), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: ask Santiago about weight initialization
        normal(self.weights, 2, 1)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.add_self_loops)
        x_out = x.unsqueeze(-1)
        op = x_out.expand(-1, -1, self.out_channels) - relu(self.weights)
        op = torch.min(op, dim=1)[0]

        out = self.propagate(edge_index=edge_index, x=op, size=None)

        if self.bias is not None:
            out = torch.max(out, self.bias)

        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """

        if ptr is not None:
            for _ in range(self.node_dim):
                ptr = ptr.unsqueeze(0)
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Delirium(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, k: int, aggr='max', add_self_loop: bool = False):
        if aggr == 'min':
            aggr = None
        super(Delirium, self).__init__(aggr=aggr)
        self.aggr = 'min' if aggr is None else aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.loop = add_self_loop
        self.nn = MLP([2 * self.in_channels, self.out_channels])
        self.morho_weights = Parameter(Tensor(self.k, self.out_channels), requires_grad=True)

    def reset_parameters(self):
        normal(self.morho_weights, 2, 1)
        reset(self.nn)

    def forward(self,
                x: Union[Tensor, PairTensor],
                batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=self.loop)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        x_out = self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        x_out = x_out.reshape(-1, self.k, self.out_channels)
        out = x_out - relu(self.morho_weights)
        return out.reshape(-1, self.out_channels)

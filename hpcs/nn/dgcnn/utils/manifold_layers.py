import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from geoopt import ManifoldParameter


class RiemannianLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold

        self._weight = Parameter(torch.Tensor(out_features, in_features))
        self.over_param = over_param
        self.weight_norm = weight_norm
        if self.over_param:
            self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
        else:
            self._bias = Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()

    @property
    def weight(self):
        return self.manifold.transp0(self.bias, self._weight)  # weight \in T_0 => weight \in T_bias

    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(self._weight * self._bias)  # reparameterisation of a point on the manifold

    def reset_parameters(self):
        init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad(): self._bias.set_(self.manifold.expmap0(self._bias))



class MobiusLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(MobiusLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        res = self.manifold.mobius_matvec(self.weight, input)
        return res

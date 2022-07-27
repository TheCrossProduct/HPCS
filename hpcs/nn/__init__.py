from torch.nn import Sequential, Linear
from torch.nn.init import zeros_


def init_weights(nn, initializer):
    if isinstance(nn, Sequential):
        for layer in nn:
            init_weights(layer, initializer)
    elif isinstance(nn, Linear):
        initializer(nn.weight)
        if nn.bias is not None:
            zeros_(nn.bias)
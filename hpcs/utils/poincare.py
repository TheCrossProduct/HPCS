import torch
from hpcs.utils.math import tanh, arctanh
"""Poincare utils functions."""

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


def egrad2rgrad(p, dp):
    """Converts Euclidean gradient to Hyperbolic gradient."""
    lambda_p = lambda_(p)
    dp /= lambda_p.pow(2)
    return dp


def lambda_(x):
    """Computes the conformal factor."""
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1. - x_sqnorm).clamp_min(MIN_NORM)


def inner(x, u, v=None):
    """Computes inner product for two tangent vectors."""
    if v is None:
        v = u
    lx = lambda_(x)
    return lx ** 2 * (u * v).sum(dim=-1, keepdim=True)


def gyration(u, v, w):
    """Gyration."""
    u2 = u.pow(2).sum(dim=-1, keepdim=True)
    v2 = v.pow(2).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)
    uw = (u * w).sum(dim=-1, keepdim=True)
    vw = (v * w).sum(dim=-1, keepdim=True)
    a = - uw * v2 + vw + 2 * uv * vw
    b = - vw * u2 - uw
    d = 1 + 2 * uv + u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def ptransp(x, y, u):
    """Parallel transport."""
    lx = lambda_(x)
    ly = lambda_(y)
    return gyration(y, -x, u) * lx / ly


def expmap_1(u, p):
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = tanh(lambda_(p) * u_norm / 2) * u / u_norm
    gamma_1 = mobius_add(p, second_term)
    return gamma_1



def expmap(p: torch.Tensor, v: torch.Tensor, r: torch.Tensor):
    """
    Computes exponential map
    """
    v_norm, p_norm = torch.norm(v), torch.norm(p)
    second_term = torch.tanh((r * v_norm) / (r**2 - p_norm**2)) * (r * v / v_norm)
    p = p.to(second_term.device)
    y = mobius_add(p, second_term)
    return y


def project(x):
    """Projects points on the manifold."""
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y):
    """Mobius addition."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def mobius_transf(z, x, pairwise=True):
    # mobius transf that maps x to origin and y to M(y)
    z1 = torch.view_as_complex(z)
    x1 = torch.view_as_complex(x)

    if not pairwise:
        if z1.dim() == 1:
            z1 = z1.view(-1, 1)
        if x1.dim() == 1:
            x1 = x1.view(1, -1)

    num = x1 - z1
    den = 1 - z1.conj()*x1
    out = num / den

    return torch.view_as_real(out)


def inverse_mobius_transf(z, x, pairwise=True):
    # inverse map of mobius transf
    z1 = torch.view_as_complex(z)
    x1 = torch.view_as_complex(x)

    if not pairwise:
        if z1.dim() == 1:
            z1 = z1.view(-1, 1)
        if x1.dim() == 1:
            x1 = x1.view(1, -1)

    num = x1 + z1
    den = 1 + z1.conj()*x1

    out = num / den

    return torch.view_as_real(out)


def mobius_mul(x, t):
    """Mobius scalar multiplication."""
    normx = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return tanh(t * arctanh(normx)) * x / normx


def get_midpoint_o(x):
    """
    Computes hyperbolic midpoint between x and the origin.
    """
    return mobius_mul(x, 0.5)


def hyp_dist_o(x):
    """
    Computes hyperbolic distance between x and the origin.
    """
    x_norm = x.norm(dim=-1, p=2, keepdim=True)
    return 2 * arctanh(x_norm)

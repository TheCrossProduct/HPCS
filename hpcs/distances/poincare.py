import torch
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from hpcs.utils.math import arctanh, tanh

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


def expmap(u, p):
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = tanh(lambda_(p) * u_norm / 2) * u / u_norm
    gamma_1 = mobius_add(p, second_term)
    return gamma_1


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


class HyperbolicDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p = 2
        self.normalize_embeddings = False
        assert not self.normalize_embeddings

    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        x = project(query_emb)
        y = project(ref_emb)

        if dtype == torch.float16:  # cdist doesn't work for float16
            rows, cols = lmu.meshgrid_from_sizes(x, y, dim=0)
            xy = torch.zeros(rows.size(), dtype=dtype).to(device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(x[rows], y[cols])
            xy[rows, cols] = distances
        else:
            xy = torch.cdist(x, y, p=self.p) ** self.p

        xx = 1 - torch.norm(x, p=self.p, dim=-1, keepdim=True) ** self.p
        yy = 1 - torch.norm(y, p=self.p, dim=-1, keepdim=True) ** self.p

        if x.dim() == 2:
            ## xx and yy have shape Nx1
            xxyy = torch.matmul(xx, yy.T)
        elif x.dim() == 3:
            ## xx and yy have shape BxNx1
            xxyy = torch.einsum('ikj,ihj->ikh', xx, yy)
        else:
            raise ValueError("Distance implemented only for tensors of rank 2 and 3")

        dxy = 1 + 2 * (xy / xxyy)

        return torch.acosh(dxy)

    def pairwise_distance(self, query_emb, ref_emb):
        x = project(query_emb)
        y = project(ref_emb)
        xy = torch.nn.functional.pairwise_distance(x, y, p=self.p) ** self.p
        xx = 1 - torch.norm(x, dim=-1, p=self.p) ** self.p
        yy = 1 - torch.norm(y, dim=-1, p=self.p) ** self.p

        dxy = 1 + 2 * (xy / (xx*yy))

        return torch.acosh(dxy)


class HyperbolicLCA(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p = 2
        self.normalize_embeddings = False
        assert not self.normalize_embeddings

    def compute_mat(self, query_emb, ref_emb):
        device = query_emb.device

        if ref_emb is None:
            ref_emb = query_emb

        # x has shape Nx2
        x = project(query_emb)
        # y has shape Mx2
        y = project(ref_emb)

        # num of points in query_emb
        n = x.shape[0]
        # num of points in ref_emb
        m = y.shape[0]

        # the output must be a matrix NxM
        dox = hyp_dist_o(x) + torch.zeros(1, m, device=device)
        doy = hyp_dist_o(y).T + torch.zeros(n, 1, device=device)
        mapd_y = mobius_transf(x, y, pairwise=False)
        mid_points = get_midpoint_o(mapd_y)
        m = inverse_mobius_transf(x, mid_points, pairwise=False)
        dom = hyp_dist_o(m)[..., 0]

        return torch.min(torch.min(dox, doy), dom)


    def pairwise_distance(self, query_emb, ref_emb):
        # query_emb and ref_emb size is NxC
        x = project(query_emb)
        y = project(ref_emb)

        mapd_y = mobius_transf(x, y)
        mid_points = get_midpoint_o(mapd_y)
        m = inverse_mobius_transf(x, mid_points)
        m_dists = hyp_dist_o(m)
        x_dists = hyp_dist_o(x)
        y_dists = hyp_dist_o(y)
        # vertex = torch.cat([x_dists, y_dists, m_dists], dim=-1)
        dists = torch.min(min(x_dists, y_dists), m_dists)
        return dists

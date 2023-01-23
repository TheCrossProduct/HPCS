"""Math utils functions."""
from typing import Union
import torch

# ################# rot 3d ########################
def yaw_rot(alpha: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(alpha, float):
        alpha = torch.Tensor([alpha])

    return torch.Tensor([
        [torch.cos(alpha), -torch.sin(alpha), 0],
        [torch.sin(alpha),  torch.cos(alpha), 0],
        [0, 0, 1],
    ])


def pitch_rot(beta: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(beta, float):
        beta = torch.Tensor([beta])

    return torch.Tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)],
    ])


def roll_rot(gamma: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(gamma, float):
        gamma = torch.Tensor([gamma])

    return torch.Tensor([
        [1, 0, 0],
        [0, torch.cos(gamma), -torch.sin(gamma)],
        [0, torch.sin(gamma),  torch.cos(gamma)],
    ])


def rot_3D(yaw: Union[float, torch.Tensor], pitch: Union[float, torch.Tensor], roll: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Function that return a 3D rotation matrix from yaw, pitch and roll rotation angles

    Parameters
    ----------
    yaw: yaw angle
    pitch: pitch angle
    roll: roll angle

    Returns
    -------
    rot_mat: torch.Tensor
    """
    R_yaw = yaw_rot(yaw)
    R_pitch = pitch_rot(pitch)
    R_roll = roll_rot(roll)
    return R_yaw @ R_pitch @ R_roll


# ################# tanh ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output / (1 - input ** 2)
        return grad


def arctanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# cosh ########################

class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1 + 1e-7)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5


def arcosh(x):
    return Arcosh.apply(x)


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


# ################# sinh ########################

class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


def arsinh(x):
    return Arsinh.apply(x)


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()

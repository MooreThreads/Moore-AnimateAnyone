import torch

def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()


tensor_interpolation = slerp


def get_tensor_interpolation_method():
    return tensor_interpolation


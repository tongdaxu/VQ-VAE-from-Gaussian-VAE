import torch
from torch import Tensor

__all__ = ["gq_cuda"]


def gq_cuda(a: Tensor, b: Tensor, c: Tensor, out: Tensor, d: int, e: int, f: int, g: float) -> None:
    torch.ops.extension_cpp.gq.default(a, b, c, out, d, e, f, g)


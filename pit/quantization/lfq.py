# From https://github.com/TencentARC/SEED-Voken/blob/main/src/Open_MAGVIT2/modules/vqvae/lookup_free_quantize.py

from math import log2, ceil
from collections import namedtuple

import torch
from torch import einsum
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, reduce


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def entropy(prob):
    return (-prob * torch.log(prob + 1e-5)).sum(dim=-1)


def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y


def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))


def lfq_entropy_loss(
    logits,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    avg_probs = reduce(probs, "... D -> D", "mean")
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss


class LFQQuantizer(Module):
    def __init__(
        self,
        format,
        codebook_size=None,
        num_codebooks=1,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
    ):
        super().__init__()

        self.format = format
        assert self.format in ["bchw", "blc"]

        self.codebook_size = codebook_size
        self.codebook_dim = int(log2(codebook_size))
        self.num_codebooks = num_codebooks

        # for entropy loss
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # for no auxiliary loss, during inference
        self.register_buffer(
            "mask", 2 ** torch.arange(self.codebook_dim), persistent=False
        )
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # codes
        all_codes = torch.arange(codebook_size)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0
        self.register_buffer("codebook", codebook, persistent=False)

    def indices_to_bits(self, x):
        """
        x: long tensor of indices

        returns big endian bits
        """
        mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        return x

    def dequant(self, x):
        return x

    def forward(
        self,
        x,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        if self.format == "bchw":
            b, c, h, w = x.shape
            ndim = c * h * w
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            b, l, c = x.shape
            ndim = l * c

        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)

        codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
        quantized = torch.where(
            x > 0, codebook_value, -codebook_value
        )  # higher than 0 filled
        quantized_flatten = rearrange(quantized, "b n c d -> b n (c d)")
        indices_v = ((quantized_flatten + 1.0) / 2.0).to(dtype=torch.long)
        indices = torch.zeros_like(quantized_flatten[:,:,0:1], dtype=torch.long)
        for i in range(quantized_flatten.shape[-1]):
            indices *= 2
            indices += indices_v[:,:,i:i+1]
        # entropy aux loss

        if self.training:
            logits = 2 * einsum("... i d, j d -> ... i j", x, self.codebook)
            # the same as euclidean distance up to a constant
            per_sample_entropy, codebook_entropy, entropy_aux_loss = lfq_entropy_loss(
                logits=logits,
                sample_minimization_weight=self.sample_minimization_weight,
                batch_maximization_weight=self.batch_maximization_weight,
            )
            avg_probs = self.zero
        else:
            # if not training, just return dummy 0
            per_sample_entropy = codebook_entropy = self.zero
            ## calculate the codebook_entropy needed for one batch evaluation
            entropy_aux_loss = self.zero
            avg_probs = self.zero

        # commit loss

        if self.training:
            commit_loss = F.mse_loss(x, quantized.detach(), reduction="none")
            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero

        # use straight-through gradients (optionally with custom activation fn) if training

        quantized = x + (quantized - x).detach()  # transfer to quantized

        # merge back codebook dim

        quantized = rearrange(quantized, "b n c d -> b n (c d)")

        # reconstitute image or video dimensions

        if self.format == "bchw":
            quantized = rearrange(quantized, "b (h w) c -> b c h w", h=h)
            indices = rearrange(indices, "b (h w) c -> b c h w", h=h)

        info = {
            "indices": indices,
            "entropy_aux_loss": entropy_aux_loss,
            "per_sample_entropy": per_sample_entropy.detach(),
            "codebook_entropy": codebook_entropy.detach(),
            "commit_loss": commit_loss,
        }
        return quantized, info

    def dequant(self, indices):
        if self.format == "bchw":
            b, ng, h, w = indices.shape
            l = h * w
            indices = rearrange(indices, "b c h w -> b (h w) c")
        else:
            b, l, ng = indices.shape
        c = self.num_codebooks * self.codebook_dim
        # b l ng, incides -> b l 16 ng value...
        quantized = torch.zeros(
            [b, l, ng, c], device=indices.device, dtype=torch.float32
        )
        for i in range(c):
            quantized[:, :, :, 15 - i] = (indices % 2).to(dtype=torch.float32)
            indices = indices // 2

        quantized = quantized * 2.0 - 1.0  # convert to [-1, 1] range
        if self.format == "bchw":
            quantized = rearrange(quantized, "b (h w) c n -> b (c n) h w", h=h)

        return quantized

if __name__ == "__main__":
    # Example usage
    lfq = LFQQuantizer(format="bchw", codebook_size=256, num_codebooks=2)
    z = torch.randn(2, 16, 32, 32)  # Example input tensor
    z_q, info = lfq(z)
    print(info["indices"])
    z_q2 = lfq.dequant(info["indices"])
    print("error", torch.mean(torch.abs(z_q - z_q2)))
    print("Quantized shape:", z_q.shape)
    print("Info:", info.keys())

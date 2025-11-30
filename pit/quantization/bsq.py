from math import log2, ceil
from collections import namedtuple

import torch
from torch import einsum
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, reduce

from pit.quantization.lfq import LFQQuantizer, lfq_entropy_loss


def bsq_entropy_loss(
    x,
    embed_dim,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    probs = torch.sigmoid(-4 * x / (embed_dim**0.5) / temperature)
    probs = torch.stack([probs, 1 - probs], dim=-1)

    log_probs = torch.log(probs + eps)

    avg_probs = reduce(probs, "... g d -> g d", "mean")
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, [-2, -1])
    sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss


class BSQQuantizer(LFQQuantizer):
    """
    BSQQuantizer is a variant of LFQQuantizer that uses a different quantization scheme.
    It is designed to work with the BSQ format, which is a block-based quantization format.
    """

    def __init__(
        self,
        format,
        codebook_size,
        num_codebooks=1,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
    ):
        super().__init__(
            format=format,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
        )
        assert self.format in ["bchw", "blc"]
        self.embed_dim = self.codebook_dim * num_codebooks

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

        # x in blc format, normalize in c dimension
        x = F.normalize(x, dim=-1)

        q_scale = 1.0 / (self.embed_dim**0.5)

        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)

        codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
        quantized = torch.where(
            x > 0, codebook_value, -codebook_value
        )  # higher than 0 filled
        indices_v = ((quantized + 1.0) / 2.0).to(dtype=torch.long)
        indices = torch.zeros_like(indices_v[:,:,0,:], dtype=torch.long)
        for i in range(16):
            indices *= 2
            indices += indices_v[:,:,i,:]

        quantized = x + (quantized - x).detach()  # transfer to quantized
        quantized = quantized * q_scale  # scale the quantized values

        if self.training:
            # the same as euclidean distance up to a constant
            per_sample_entropy, codebook_entropy, entropy_aux_loss = bsq_entropy_loss(
                x=x,
                embed_dim=self.embed_dim,
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

        # merge back codebook dim
        quantized = rearrange(quantized, "b n c d -> b n (c d)")

        if self.format == "bchw":
            quantized = rearrange(quantized, "b (h w) c -> b c h w", h=h)
            indices = rearrange(indices, "b (h w) c -> b c h w", h=h)

        info = {
            "indices": indices,
            "entropy_aux_loss": entropy_aux_loss,
            "per_sample_entropy": per_sample_entropy.detach(),
            "codebook_entropy": codebook_entropy.detach(),
        }
        return quantized, info

    def dequant(self, indices):
        if self.format == "bchw":
            b, ng, h, w = indices.shape
            l = h * w
            indices = rearrange(indices, "b c h w -> b (h w) c")
        else:
            b, l, ng = indices.shape
        # b l ng, incides -> b l 16 ng value...
        quantized = torch.zeros(
            [b, l, ng, 16], device=indices.device, dtype=torch.float32
        )
        for i in range(16):
            quantized[:, :, :, 15 - i] = (indices % 2).to(dtype=torch.float32)
            indices = indices // 2
        q_scale = 1.0 / (self.embed_dim**0.5)

        quantized = quantized * 2.0 - 1.0  # convert to [-1, 1] range
        quantized *= q_scale
        if self.format == "bchw":
            quantized = rearrange(quantized, "b (h w) c n -> b (c n) h w", h=h)

        return quantized

if __name__ == "__main__":
    quantizer = BSQQuantizer(
        format="bchw",
        codebook_size=2,  # codebook size, must be a power of 2
        num_codebooks=16,
        sample_minimization_weight=1.0,  # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        batch_maximization_weight=1.0,
    )

    image_feats = torch.randn(
        2, 16, 32, 32
    )  # 16 is dim, must be power of 2 of codebook_size

    quantized, info = quantizer(
        image_feats
    )  # you may want to experiment with temperature
    quantized_2 = quantizer.dequant(info["indices"])
    print("indices shape:", info["indices"].shape)
    print("quantized shape:", quantized.shape)
    print("err", torch.mean(torch.abs(quantized - quantized_2)))

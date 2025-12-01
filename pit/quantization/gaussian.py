import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions import Normal
from torch.quasirandom import SobolEngine
from scipy.stats import norm
import math

try:
    import gq_cuda
    HAS_GQ_CUDA = True
except ImportError:
    HAS_GQ_CUDA = False

def prior_samples(n_samples, n_variable, seed_rec):
    sobol = SobolEngine(n_variable, scramble=True, seed=seed_rec)
    samples_sobol = sobol.draw(n_samples)
    samples_i = torch.from_numpy(norm.ppf(samples_sobol))
    return samples_i

class GaussianQuantRegularizer(nn.Module):
    # GaussianQuant Regularizer
    # train() mode: sample as Gaussian VAE, log contains loss function
    # eval() mode: sample as VQ-VAE, log contains indices
    def __init__(self, format, n_samples, group=1, logvar_range=[-30.0, 20.0], # common parameters
                 tolerance = 0.5, lam_factor=1.01, # train parameters
                 seed=42, beta=1.0, backend="torch", # inference parameters
        ):
        super().__init__()

        self.format = format
        assert self.format in ["bchw", "blc"]
        self.group = group
        self.logvar_range = logvar_range
        self.n_samples = n_samples
        self.log_n_samples = int(math.log(self.n_samples, 2))

        # training parameter setup
        self.lam_factor = lam_factor
        self.lam = 1.0
        self.lam_min = 1.0
        self.lam_max = 1.0
        self.lam_range = (1e-3, 1e3)
        self.tolerance = tolerance

        # inference parameter setup
        self.beta = beta
        self.seed = seed
        self.register_buffer("prior_samples", prior_samples(self.n_samples, self.group, self.seed).float(), persistent=False)
        self.normal_dist = Normal(torch.zeros([1, self.group]), torch.ones([1, self.group]))
        self.register_buffer("normal_log_prob", self.normal_dist.log_prob(self.prior_samples).float(), persistent=False)

        self.logvar_range = logvar_range
        self.perturbed = None
        if backend == "cuda" and HAS_GQ_CUDA is False:
            print("no gq cuda module is detected, use pytorch backend!")
            backend = "torch"
        self.backend = backend

    def forward(self, z):
        z = z.float()

        if self.format == "bchw":
            b, c, h, w = z.shape
            c = c // 2
            ndim = c * h * w
            l = h * w
            zhat = rearrange(z, "b c h w -> b (h w) c")
        else:
            b, l, c = z.shape
            c = c // 2
            ndim = l * c
            zhat = z

        # b, l, c
        mu, logvar = zhat.chunk(2, 2)
        logvar = torch.clamp(logvar, self.logvar_range[0], self.logvar_range[1])
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)

        ## train time behaviour
        if self.training:
            zhat = mu + torch.randn_like(mu) * std
            # blc
            kl2 = 1.4426 * 0.5 * (torch.pow(mu, 2) + var - 1.0 - logvar)
            kl2 = kl2.reshape(b,l,self.group,c//self.group)
            kl2 = torch.sum(kl2,dim=2) # sum over group dimension
            kl2_mean, kl2_min, kl2_max = torch.mean(kl2), torch.min(kl2), torch.max(kl2)

            ge = (kl2 > self.log_n_samples + self.tolerance).type(kl2.dtype) * self.lam_max
            eq = (kl2 <= self.log_n_samples + self.tolerance).type(kl2.dtype) * (
                kl2 >= self.log_n_samples - self.tolerance
            ).type(kl2.dtype)
            le = (kl2 < self.log_n_samples - self.tolerance).type(kl2.dtype) * self.lam_min
            kl_loss = torch.sum((ge * kl2 + eq * kl2 + le * kl2), dim=[1,2])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            # update lambda
            if kl2_mean > self.log_n_samples:
                self.lam = self.lam * self.lam_factor
            else:
                self.lam = self.lam / self.lam_factor

            if kl2_max > self.log_n_samples + self.tolerance:
                self.lam_max = self.lam_max * self.lam_factor
            else:
                self.lam_max / self.lam_max * self.lam_factor
            self.lam_max = max(min(self.lam_max, self.lam_range[1]), 1.0)
            if kl2_min < self.log_n_samples - self.tolerance:
                self.lam_min = self.lam_min / self.lam_factor
            else:
                self.lam_min = self.lam_min * self.lam_factor
            self.lam_min = max(min(self.lam_min, 1.0), self.lam_range[0])

            if self.format == "bchw":
                zhat = rearrange(zhat, "b (h w) c -> b c h w", h=h)
            info = {"kl_loss": torch.mean(kl_loss), "bits-mean": kl2_mean, "bits-min": kl2_min, "bits-max": kl2_max}
        else:
            zhat_noquant = mu + torch.randn_like(mu) * std
            mu = mu.reshape(b, l, self.group, c // self.group).permute(0,1,3,2).reshape(-1, self.group)
            std = std.reshape(b, l, self.group, c // self.group).permute(0,1,3,2).reshape(-1, self.group)
            if self.backend == "cuda":
                # cuda impl
                if self.perturbed is None or self.perturbed.shape[0] != mu.shape[0]:
                    # shape change, need buffer update
                    self.perturbed = torch.zeros([mu.shape[0], self.n_samples]).to(device=mu.device).contiguous()
                gq_cuda.ops.gq_cuda(
                    mu,std,self.prior_samples,self.perturbed,self.group,mu.shape[0],self.n_samples,self.beta
                )
                indices = torch.argmax(self.perturbed, dim=1).clone()
                zhat = torch.index_select(self.prior_samples, 0, indices)
            elif self.backend == "torch":
                # torch impl
                bs = mu.shape[0] // 8
                zhat = torch.zeros_like(mu)
                indices = torch.zeros([mu.shape[0]], device=mu.device, dtype=torch.long)
                for i in range(0, mu.shape[0], bs):
                    mu_q = mu[i : i + bs]
                    std_q = std[i:i+bs]
                    q_normal_dist = Normal(mu_q[:, None, :], std_q[:, None, :])
                    log_ratios = (
                        q_normal_dist.log_prob(self.prior_samples[None])
                        - self.normal_log_prob[None] * self.beta
                    )
                    perturbed = torch.sum(log_ratios, dim=2)
                    argmax_indices = torch.argmax(perturbed, dim=1)
                    zhat[i : i + bs] = torch.index_select(self.prior_samples, 0, argmax_indices)
                    indices[i : i + bs] = argmax_indices
            else:
                raise ValueError
            zhat = zhat.reshape(b, l, c // self.group, self.group).permute(0, 1, 3, 2).reshape(b, l, c).float()
            indices = indices.reshape(b, l, c // self.group)
            if self.format == "bchw":
                zhat = rearrange(zhat, "b (h w) c -> b c h w", h=h)
                zhat_noquant = rearrange(zhat_noquant, "b (h w) c -> b c h w", h=h)
                indices = rearrange(indices, "b (h w) c -> b c h w", h=h)
            info = {"indices": indices, "zhat_noquant": zhat_noquant, }
        return zhat, info

    def dequant(self, indices):
        if self.format == "bchw":
            b, ng, h, w = indices.shape
            l = h * w
            indices = rearrange(indices, "b c h w -> b (h w) c")
        else:
            b, l, ng = indices.shape
            # here, c is number of groups

        indices = indices.reshape(-1)
        zhat = torch.zeros([b*l*ng, self.group], device=indices.device, dtype=torch.float32)
        zhat = torch.index_select(self.prior_samples, 0, indices).float()
        zhat = zhat.reshape(b, l, ng, self.group).permute(0, 1, 3, 2).reshape(b, l, ng * self.group)

        if self.format == "bchw":
            zhat = rearrange(zhat, "b (h w) c -> b c h w", h=h)
        return zhat


class IdentityRegularizer(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, z):
        return z, dict()


if __name__ == "__main__":
    z = torch.randn([1, 32, 4, 4]).cuda()
    gauss = GaussianQuantRegularizer(format="bchw", group=16, n_samples=1024, seed=42).cuda()
    zhat, info = gauss(z)
    z2 = gauss.dequant(info["indices"])

    print(zhat.shape, z2.shape)

    print(torch.mean(torch.abs(zhat - z2)))

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


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

class GaussianRegularizer(nn.Module):
    # Gaussian VAE

    # args
    # levels: the FSQ levels parameter, see Table 1 of FSQ paper
    # format: data format, must be one of []"bchw", "blc"]

    def __init__(self, format, logvar_range=[-30.0, 20.0], group=1, mode="even", target=16):
        super().__init__()

        self.format = format

        assert self.format in ["bchw", "blc"]
        self.logvar_range = logvar_range
        self.group = group
        self.mode = mode
        self.target = target

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

        zhat = mu + torch.randn_like(mu) * std

        # blc
        kl2 = 1.4426 * 0.5 * (torch.pow(mu, 2) + var - 1.0 - logvar)
        kl2 = kl2.reshape(b,l,self.group,c//self.group)
        kl2 = torch.sum(kl2,dim=2) # sum over group dimension
        kl2_mean, kl2_min, kl2_max = torch.mean(kl2), torch.min(kl2), torch.max(kl2)

        if self.mode == "even":
            kl_loss = 0.5 * torch.sum(
                torch.pow(mu, 2) + var - 1.0 - logvar,
                dim=[1, 2],
            )
        elif self.mode == "target":
            FLIP_FACTOR = 40
            TOLERANCE = 0.35
            TARGET = self.target / 1.4426 # bits to nats
            kl_loss = 0.5 * (torch.pow(mu, 2) + var - 1.0 - logvar)
            kl_loss = kl_loss.reshape(b,l,self.group,c//self.group)
            kl_loss = torch.sum(kl_loss,dim=2) # sum over group dimension
            dtype = kl_loss.dtype
            ge = (kl_loss > TARGET + TOLERANCE).type(dtype) * FLIP_FACTOR
            eq = (kl_loss <= TARGET + TOLERANCE).type(dtype) * (
                  kl_loss >= TARGET - TOLERANCE
            ).type(dtype)
            le = (kl_loss < TARGET - TOLERANCE).type(dtype) * (1 / FLIP_FACTOR)
            kl_loss = torch.sum(ge * kl_loss + eq * kl_loss + le * kl_loss, dim=[1, 2])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.format == "bchw":
            zhat = rearrange(zhat, "b (h w) c -> b c h w", h=h)

        info = {"kl": torch.mean(kl_loss), "kl2_mean": kl2_mean, "kl2_min": kl2_min, "kl2_max": kl2_max}
        return zhat, info


class GaussianQuantTrainRegularizer(nn.Module):
    def __init__(self, format, n_samples, group, tolerance = 0.5, lam_factor=1.01, sample: bool = True):
        super().__init__()
        assert(format == "bchw")
        self.group = group
        self.target = int(math.log(n_samples, 2))
        self.lam_factor = lam_factor
        self.lam = 1.0
        self.lam_min = 1.0
        self.lam_max = 1.0
        self.lam_range = (1e-3, 1e3)
        self.tolerance = tolerance

    def forward(self, z):
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.sample()

        log["mean"] = posterior.mean.detach()
        log["logvar"] = posterior.logvar.detach()
        log["std"] = posterior.std.detach()
        log["var"] = posterior.var.detach()
        log["lam_mean"] = self.lam

        kls = 1.4426 * (
            0.5
            * (torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar)
        )

        b, c, h, w = kls.shape

        kls = torch.sum(kls.reshape(b, self.group, c // self.group, h, w), dim=1)

        ge = (kls > self.target + self.tolerance).type(kls.dtype) * self.lam_max
        eq = (kls <= self.target + self.tolerance).type(kls.dtype) * (
            kls >= self.target - self.tolerance
        ).type(kls.dtype)
        le = (kls < self.target - self.tolerance).type(kls.dtype) * self.lam_min
        kl_loss = torch.sum((ge * kls + eq * kls + le * kls), dim=[1,2,3])

        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss * self.lam

        # update lambda
        if torch.mean(kls) > self.target:
            self.lam = self.lam * self.lam_factor
        else:
            self.lam = self.lam / self.lam_factor

        if torch.max(kls) > self.target + self.tolerance:
            self.lam_max = self.lam_max * self.lam_factor
        else:
            self.lam_max / self.lam_max * self.lam_factor
        self.lam_max = max(min(self.lam_max, self.lam_range[1]), 1.0)

        if torch.min(kls) < self.target - self.tolerance:
            self.lam_min = self.lam_min / self.lam_factor
        else:
            self.lam_min = self.lam_min * self.lam_factor
        self.lam_min = max(min(self.lam_min, 1.0), self.lam_range[0])

        log["bits-mean"] = torch.mean(kls).detach()
        log["bits-max"] = torch.max(kls).detach()
        log["bits-min"] = torch.min(kls).detach()

        return z, log

class IdentityRegularizer(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, z):
        return z, dict()


class GaussianQuantRegularizer(nn.Module):

    def __init__(self, format, group, n_samples, seed, beta=1.0, logvar_range=[-30.0, 20.0], backend="torch"):
        super().__init__()

        self.format = format
        assert self.format in ["bchw", "blc"]

        self.group = group
        self.n_samples = n_samples
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
            l = h * w
            z = rearrange(z, "b c h w -> b (h w) c")
            c = c // 2
        else:
            b, l, c = z.shape
            z = z
            c = c // 2

        # b, l, c
        mu, logvar = z.chunk(2, 2)
        logvar = torch.clamp(logvar, self.logvar_range[0], self.logvar_range[1])
        std = torch.exp(logvar * 0.5)

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
            indices = torch.argmax(self.perturbed, dim=1)
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
        elif self.backend == "raw":

            bs = mu.shape[0] // 8
            zhat = torch.zeros_like(mu)
            indices = torch.zeros([mu.shape[0]], device=mu.device, dtype=torch.long)

            for i in range(0, mu.shape[0], bs):
                mu_q = mu[i : i + bs]
                std_q = std[i:i+bs]

                log_ratios = - ((self.prior_samples[None] - mu_q[:, None, :]) / std_q[:, None, :]) ** 2 + (self.prior_samples[None]) ** 2 * self.beta

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
            indices = rearrange(indices, "b (h w) c -> b c h w", h=h)

        return zhat, {"indices": indices}

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

        # bs = indices.shape[0] // 8
        # for i in range(0, indices.shape[0], bs):
        #     zhat[i:i+bs] = torch.index_select(self.prior_samples, 0, indices[i:i+bs]).float()
        zhat = torch.index_select(self.prior_samples, 0, indices).float()
        zhat = zhat.reshape(b, l, ng, self.group).permute(0, 1, 3, 2).reshape(b, l, ng * self.group)

        if self.format == "bchw":
            zhat = rearrange(zhat, "b (h w) c -> b c h w", h=h)
        return zhat


if __name__ == "__main__":
    z = torch.randn([1, 32, 4, 4]).cuda()
    gauss = GaussianQuantRegularizer("bchw", 16, 1024, 42).cuda()
    zhat, info = gauss(z)
    z2 = gauss.dequant(info["indices"])

    print(zhat.shape, z2.shape)

    print(torch.mean(torch.abs(zhat - z2)))

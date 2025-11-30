import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pit.util import (
    default,
    instantiate_from_config,
    get_obj_from_str,
)


class AutoencodingPostEngine(pl.LightningModule):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        input_key,
        eval_only: bool = False,
        encoder_config: Dict,
        decoder_config: Dict,
        post_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Union[Dict, None] = None,
        ckpt_engine: Union[None, str, dict] = None,
        ckpt_path: Optional[str] = None,
        additional_decode_keys: Optional[List[str]] = None,
        clamp_range = None,
        num_flow_steps = 50,
        mmse_noise_std = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_key = input_key
        self.automatic_optimization = False  # pytorch lightning

        self.encoder: nn.Module = instantiate_from_config(encoder_config)
        self.decoder: nn.Module = instantiate_from_config(decoder_config)
        self.poster = instantiate_from_config(post_config)
        self.regularization: nn.Module = instantiate_from_config(regularizer_config)
        self.clamp_range = clamp_range
        self.eps = 0.0
        self.num_flow_steps = num_flow_steps
        self.mmse_noise_std = mmse_noise_std

        if not eval_only:
            self.optimizer_config = default(
                optimizer_config, {"target": "torch.optim.Adam"}
            )

        if ckpt_path is not None:
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        self.additional_decode_keys = set(default(additional_decode_keys, []))

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:
            return
        self.init_from_ckpt(ckpt)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)
        print(f"Restored from {path}")

    def get_input(self, batch: Dict) -> torch.Tensor:
        return batch[self.input_key]

    @torch.no_grad()
    def create_xhat_0(self, xhat):
        with torch.no_grad():
            xhat_0 = xhat + torch.randn_like(xhat) * self.mmse_noise_std
        return xhat_0

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        if unregularized:
            return z, dict()

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def quant(self, x):
        z, reg_log = self.encode(x, return_reg_log=True)
        return z, reg_log["indices"]

    def dequant(self, incides):
        zhat = self.regularization.dequant(incides)
        xhat = self.decode(zhat)
        if self.clamp_range is not None:
            xhat = torch.clamp(xhat, self.clamp_range[0], self.clamp_range[1])
        return xhat
    
    @torch.no_grad()
    def post(self, xhat):
        xhat_0 = self.create_xhat_0(xhat)
        dt = (1.0 / self.num_flow_steps) * (1.0 - self.eps)
        x_t_next = xhat_0.clone()
        x_t_seq = [x_t_next]
        t_one = torch.ones(xhat_0.shape[0], device=xhat_0.device)
        for i in range(self.num_flow_steps):
            num_t = (i / self.num_flow_steps) * (1.0 - self.eps) + self.eps
            v_t_next = self(x_t=x_t_next, t=t_one * num_t)
            x_t_next = x_t_next.clone() + v_t_next * dt
            x_t_seq.append(x_t_next)
        xhat_post = x_t_seq[-1]
        if self.clamp_range is not None:
            xhat_post = torch.clamp(xhat_post, self.clamp_range[0], self.clamp_range[1])
        return xhat_post

    def forward(self, xhat_t: torch.Tensor, t):
        return self.poster(xhat_t, t)
    
    def generate_random_t(self, bs):
        return torch.rand(bs, 1, 1, 1, device=self.device) * (1.0 - self.eps) + self.eps

    def training_step(self, batch: dict, batch_idx: int):

        x = self.get_input(batch)
        opt = self.optimizers()

        with torch.no_grad():
            z = self.encode(x)
            xhat = self.decode(z)
            t = self.generate_random_t(x.shape[0]).to(xhat.device)
            xhat_0 = self.create_xhat_0(xhat)
            xhat_t = t * x + (1.0 - t) * xhat_0

        opt.zero_grad()
        v_t = self(xhat_t, t.squeeze())
        loss = F.mse_loss(v_t, x - xhat_0)
        self.manual_backward(loss)
        opt.step()

        self.log("train/loss", loss.detach())


    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        x = self.get_input(batch)
        z = self.encode(x)
        xhat = self.decode(z)
        with torch.no_grad():
            z = self.encode(x)
            xhat = self.decode(z)
            t = self.generate_random_t(x.shape[0]).to(xhat.device)
            xhat_0 = self.create_xhat_0(xhat)
            xhat_t = t * x + (1.0 - t) * xhat_0
        v_t = self(xhat_t, t.squeeze())
        loss = F.mse_loss(v_t, x - xhat_0)
        self.log("val/loss", loss.detach())
        return {}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = self.instantiate_optimizer_from_config(
            self.poster.parameters(),
            self.learning_rate,
            self.optimizer_config,
        )
        return opt

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )
        z = self.encode(x)
        xhat = self.decode(z)
        xhat_post = self.post(xhat)

        log["inputs"] = x
        log["xhat"] = xhat
        log["xhat_post"] = xhat_post

        return log

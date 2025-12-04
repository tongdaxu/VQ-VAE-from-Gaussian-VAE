import re
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pit.modules.flux.util import load_flow_model2, load_flow_model_control
from pit.modules.flux.xflux_pipeline import XFluxPipelineClean
from diffusers import FluxPriorReduxPipeline
from pit.modules.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor
from pit.modules.flux.util import (load_ae, load_controlnet, load_checkpoint)
from pit.models.hyvae import HunyuanVAE2D
from pit.models.foundation_models import aux_foundation_model
from pit.util import (
    default,
    instantiate_from_config,
    get_obj_from_str,
)


class AutoencoderKLQwenImage(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
        self.model = AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")

    def encode(self, x,
        return_reg_log = False,
        unregularized = False,
    ):
        qzx = self.model.encode(x[:,:,None],return_dict=False)
        z = qzx[0].sample()[:,:,0]
        return z, {}

    def decode(self, z):
        xhat = self.model.decode(z[:,:,None],return_dict=False)[0][:,:,0]
        return xhat

class AutoencoderKLWAN(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
        self.model = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae")

    def encode(self, x,
        return_reg_log = False,
        unregularized = False,
    ):
        qzx = self.model.encode(x[:,:,None],return_dict=False)
        z = qzx[0].sample()[:,:,0]
        return z, {}

    def decode(self, z):
        xhat = self.model.decode(z[:,:,None],return_dict=False)[0][:,:,0]
        return xhat


class AutoencoderKLFLUX(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        from diffusers import AutoencoderKL
        self.model = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")

    def encode(self, x,
        return_reg_log = False,
        unregularized = False,
    ):
        qzx = self.model.encode(x,return_dict=False)[0]
        z = qzx.sample()
        return z, {}

    def decode(self, z):
        xhat = self.model.decode(z,return_dict=False)[0]
        return xhat


class AutoencoderKLHYImage2(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.model = HunyuanVAE2D(
            block_out_channels=[128,
                256,
                512,
                512,
                1024,
                1024
            ],
            in_channels=3,
            out_channels=3,
            latent_channels=64,
            layers_per_block=2,
            ffactor_spatial=32,
            sample_size=384,
            sample_tsize=96,
            scaling_factor=0.75289,
            downsample_match_channel=True,
            upsample_match_channel=True,
        )
        ckpt = torch.load("",map_location='cpu')["state_dict"]
        vae_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("vae."):
                vae_ckpt[k.replace("vae.", "")] = v

        self.model.load_state_dict(
            vae_ckpt
        )

    def encode(self, x,
        return_reg_log = False,
        unregularized = False,
    ):
        qzx = self.model.encode(x,return_dict=False)[0]
        z = qzx.sample()
        return z, {}

    def decode(self, z):
        xhat = self.model.decode(z,return_dict=False)[0]
        return xhat

class AutoencoderKLSD3(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        from diffusers import AutoencoderKL
        self.model = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")

    def encode(self, x,
        return_reg_log = False,
        unregularized = False,
    ):
        qzx = self.model.encode(x,return_dict=False)[0]
        z = qzx.sample()
        return z, {}

    def decode(self, z):
        xhat = self.model.decode(z,return_dict=False)[0]
        return xhat


class AutoencoderKLEQ(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        from diffusers import AutoencoderKL
        self.model = AutoencoderKL.from_pretrained("zelaki/eq-vae")

    def encode(self, x,
        return_reg_log = False,
        unregularized = False,
    ):
        qzx = self.model.encode(x,return_dict=False)[0]
        z = qzx.sample()
        return z, {}

    def decode(self, z):
        xhat = self.model.decode(z,return_dict=False)[0]
        return xhat

class AutoencoderKLHYImage3(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        import sys
        from safetensors import safe_open
        sys.path.append('')
        from autoencoder_kl_3d import AutoencoderKLConv3D
        kwargs =  {
                "block_out_channels": [
                    128,
                    256,
                    512,
                    1024,
                    1024
                ],
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 32,
                "layers_per_block": 2,
                "ffactor_spatial": 16,
                "ffactor_temporal": 4,
                "sample_size": 384,
                "sample_tsize": 96,
                "downsample_match_channel": True,
                "upsample_match_channel": True,
                "scaling_factor": 0.562679178327931
            }
        model = AutoencoderKLConv3D(
            **kwargs
        )
        tensors={}
        with safe_open("", framework="pt", device="cpu") as f:
            for key in f.keys():
                if "vae." in key:
                    tensors[key[4:]] = f.get_tensor(key)
        with safe_open("", framework="pt", device="cpu") as f:
            for key in f.keys():
                if "vae." in key:
                    tensors[key[4:]] = f.get_tensor(key)

        miss, unexp = model.load_state_dict(tensors, strict=False)
        print(miss, unexp)

        self.model = model

    def encode(self, x,
        return_reg_log = False,
        unregularized = False,
    ):
        qzx = self.model.encode(x[:,:,None],return_dict=False)
        z = qzx[0].sample()[:,:,0]
        return z, {}

    def decode(self, z):
        xhat = self.model.decode(z[:,:,None],return_dict=False)[0][:,:,0]
        return xhat


class AutoencodingEngine(pl.LightningModule):
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
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        trainable_ae_params: Optional[List[List[str]]] = None,
        ae_optimizer_args: Optional[List[dict]] = None,
        trainable_disc_params: Optional[List[List[str]]] = None,
        disc_optimizer_args: Optional[List[dict]] = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3.0,
        ckpt_engine: Union[None, str, dict] = None,
        ckpt_path: Optional[str] = None,
        additional_decode_keys: Optional[List[str]] = None,
        use_vf = None,
        reverse_proj = False,
        clamp_range = None,
        latent_stats = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_key = input_key
        self.automatic_optimization = False  # pytorch lightning

        self.encoder: nn.Module = instantiate_from_config(encoder_config)
        self.decoder: nn.Module = instantiate_from_config(decoder_config)
        self.regularization: nn.Module = instantiate_from_config(regularizer_config)
        self.clamp_range = clamp_range
        self.latent_stats = latent_stats
        if self.latent_stats:
            self.latent_mean = nn.Parameter(torch.zeros([1,encoder_config.params.z_channels,1,1]), requires_grad=False)
            self.latent_std = nn.Parameter(torch.zeros([1,encoder_config.params.z_channels,1,1]), requires_grad=False)

        if use_vf is not None:
            self.use_vf = use_vf
            print(f"Using {use_vf} as auxiliary feature.")
            self.foundation_model = aux_foundation_model(use_vf)
            vf_feature_dim = self.foundation_model.feature_dim
            embed_dim = encoder_config["params"]["z_channels"]
            self.linear_proj = torch.nn.Conv2d(vf_feature_dim, embed_dim, kernel_size=1, bias=True)
            if reverse_proj:
                self.linear_proj = torch.nn.Conv2d(embed_dim, vf_feature_dim, kernel_size=1, bias=False)
        else:
            self.use_vf = None
        self.reverse_proj = reverse_proj

        if not eval_only:
            
            self.loss: nn.Module = instantiate_from_config(loss_config)
            self.optimizer_config = default(
                optimizer_config, {"target": "torch.optim.Adam"}
            )
            self.diff_boost_factor = diff_boost_factor
            self.disc_start_iter = disc_start_iter
            self.lr_g_factor = lr_g_factor
            self.trainable_ae_params = trainable_ae_params
            if self.trainable_ae_params is not None:
                self.ae_optimizer_args = default(
                    ae_optimizer_args,
                    [{} for _ in range(len(self.trainable_ae_params))],
                )
                assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
            else:
                self.ae_optimizer_args = [{}]  # makes type consitent

            self.trainable_disc_params = trainable_disc_params
            if self.trainable_disc_params is not None:
                self.disc_optimizer_args = default(
                    disc_optimizer_args,
                    [{} for _ in range(len(self.trainable_disc_params))],
                )
                assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
            else:
                self.disc_optimizer_args = [{}]  # makes type consitent

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
        # print("Unexpected keys: ", unexpected_keys)
        print(f"Restored from {path}")

    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first
        # format (e.g., bchw instead if bhwc)
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + list(self.encoder.parameters())
        params = params + list(self.decoder.parameters())
        if self.use_vf is not None:
            params = params + list(self.linear_proj.parameters())
        return params

    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            params = []
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

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

        if self.latent_stats:
            z = (z - self.latent_mean) / self.latent_std

        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:

        if self.latent_stats:
            z = z * self.latent_std + self.latent_mean

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

    def forward(
        self, x: torch.Tensor, encoder_grad=True, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        if encoder_grad:
            z, reg_log = self.encode(x, return_reg_log=True)
        else:
            with torch.no_grad():
                z, reg_log = self.encode(x, return_reg_log=True)

        dec = self.decode(z, **additional_decode_kwargs)

        if self.use_vf is not None:
            aux_feature = self.foundation_model(x)
            if not self.reverse_proj:
                aux_feature = self.linear_proj(aux_feature)
            else:
                zp = torch.nn.functional.interpolate(z, size=(aux_feature.shape[2], aux_feature.shape[3]), mode="bilinear")
                zp = self.linear_proj(zp)
            reg_log["aux_feature"] = aux_feature
            reg_log["zp"] = zp

        if self.clamp_range is not None:
            dec = torch.clamp(dec, self.clamp_range[0], self.clamp_range[1])
        return z, dec, reg_log

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            if self.use_vf:
                extra_info["enc_last_layer"] = self.encoder.conv_out.weight
            else:
                extra_info["enc_last_layer"] = None
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(x, xrec, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(
                batch, batch_idx, optimizer_idx=optimizer_idx
            )
            self.manual_backward(loss)

        opt.step()

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, xrec, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

    def get_param_groups(
        self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = []
        num_params = 0
        for names, args in zip(parameter_names, optimizer_args):
            params = []
            for pattern_ in names:
                pattern_params = []
                pattern = re.compile(pattern_)
                for p_name, param in self.named_parameters():
                    if re.match(pattern, p_name):
                        pattern_params.append(param)
                        num_params += param.numel()
                params.extend(pattern_params)
            groups.append({"params": params, **args})
        return groups, num_params

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        if self.trainable_ae_params is None:
            ae_params = self.get_autoencoder_params()
        else:
            ae_params, num_ae_params = self.get_param_groups(
                self.trainable_ae_params, self.ae_optimizer_args
            )
        if self.trainable_disc_params is None:
            disc_params = self.get_discriminator_params()
        else:
            disc_params, num_disc_params = self.get_param_groups(
                self.trainable_disc_params, self.disc_optimizer_args
            )
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opts = [opt_ae]
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(
                disc_params, self.learning_rate, self.optimizer_config
            )
            opts.append(opt_disc)

        return opts

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

        _, xrec, _ = self(x, **additional_decode_kwargs)
        log["inputs"] = x
        log["reconstructions"] = xrec
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        log["diff_boost"] = (
            2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        )
        if hasattr(self.loss, "log_images"):
            log.update(self.loss.log_images(x, xrec))

        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = xrec_add
        return log


def redux_call(
    self,
    image,
    prompt=None,
    prompt_2= None,
    prompt_embeds=None,
    pooled_prompt_embeds = None,
    prompt_embeds_scale = 1.0,
    pooled_prompt_embeds_scale = 1.0,
    return_dict: bool = True,
):
    self.check_inputs(
        image,
        prompt,
        prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        prompt_embeds_scale=prompt_embeds_scale,
        pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
    )

    # 2. Define call parameters
    if image is not None and isinstance(image, Image.Image):
        batch_size = 1
    elif image is not None and isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]
    if prompt is not None and isinstance(prompt, str):
        prompt = batch_size * [prompt]
    if isinstance(prompt_embeds_scale, float):
        prompt_embeds_scale = batch_size * [prompt_embeds_scale]
    if isinstance(pooled_prompt_embeds_scale, float):
        pooled_prompt_embeds_scale = batch_size * [pooled_prompt_embeds_scale]

    device = self._execution_device

    # 3. Prepare image embeddings
    image_latents = self.encode_image(image, device, 1)

    image_embeds = self.image_embedder(image_latents).image_embeds
    image_embeds = image_embeds.to(device=device)

    # 3. Prepare (dummy) text embeddings
    if hasattr(self, "text_encoder") and self.text_encoder is not None:
        (
            prompt_embeds,
            pooled_prompt_embeds,
            _,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )
    else:
        # max_sequence_length is 512, t5 encoder hidden size is 4096
        prompt_embeds = torch.zeros((batch_size, 512, 4096), device=device, dtype=image_embeds.dtype)
        # pooled_prompt_embeds is 768, clip text encoder hidden size
        pooled_prompt_embeds = torch.zeros((batch_size, 768), device=device, dtype=image_embeds.dtype)

    # scale & concatenate image and text embeddings
    prompt_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)

    prompt_embeds *= torch.tensor(prompt_embeds_scale, device=device, dtype=image_embeds.dtype)[:, None, None]
    pooled_prompt_embeds *= torch.tensor(pooled_prompt_embeds_scale, device=device, dtype=image_embeds.dtype)[
        :, None
    ]

    # Offload all models
    self.maybe_free_model_hooks()

    return (prompt_embeds, pooled_prompt_embeds)

class AutoencodingFluxEngine(AutoencodingEngine):

    def __init__(self, *args, input_key, controlnet_path, lora_path, eval_only = False, encoder_config, decoder_config, loss_config, regularizer_config, optimizer_config = None, lr_g_factor = 1, trainable_ae_params = None, ae_optimizer_args = None, trainable_disc_params = None, disc_optimizer_args = None, disc_start_iter = 0, diff_boost_factor = 3, ckpt_engine = None, ckpt_path = None, additional_decode_keys = None, clamp_range=None, **kwargs):
        super().__init__(*args, input_key=input_key, eval_only=eval_only, encoder_config=encoder_config, decoder_config=decoder_config, loss_config=loss_config, regularizer_config=regularizer_config, optimizer_config=optimizer_config, lr_g_factor=lr_g_factor, trainable_ae_params=trainable_ae_params, ae_optimizer_args=ae_optimizer_args, trainable_disc_params=trainable_disc_params, disc_optimizer_args=disc_optimizer_args, disc_start_iter=disc_start_iter, diff_boost_factor=diff_boost_factor, ckpt_engine=ckpt_engine, ckpt_path=ckpt_path, additional_decode_keys=additional_decode_keys, clamp_range=clamp_range, **kwargs)
        self.controlnet_path = controlnet_path
        self.lora_path = lora_path
        self.control_channels = encoder_config.params.z_channels

    def load_flux_pipeline(
            self, 
        ):
        device = next(self.parameters()).device
        self.ae_dtype = next(self.parameters()).dtype
        dtype = torch.bfloat16
        self.xflux_pipeline = XFluxPipelineClean("flux-dev", device, dtype)
        flux_dit = load_flow_model2(self.xflux_pipeline.model_type, device=device)

        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in flux_dit.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            if name.startswith("double_blocks") and layer_index in double_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                dim=3072, rank=128
                )
            elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                dim=3072, rank=128
                )
            else:
                lora_attn_procs[name] = attn_processor
        flux_dit.set_attn_processor(lora_attn_procs)
        flux_dit.load_state_dict(torch.load(self.lora_path, map_location='cpu'), strict=False)
        flux_dit = flux_dit.to(device=device, dtype=dtype)
        vae = load_ae(self.xflux_pipeline.model_type, device=device)
        controlnet = load_controlnet(self.xflux_pipeline.model_type, device, self.control_channels).to(dtype)
        controlnet.load_state_dict(load_checkpoint(self.controlnet_path, None, None), strict=True)
        self.redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=dtype).to(device)
        self.xflux_pipeline.model = flux_dit
        self.xflux_pipeline.ae = vae
        self.xflux_pipeline.controlnet = controlnet
        self.xflux_pipeline.controlnet_loaded = True

    def dequant(self, incides):
        zhat = self.regularization.dequant(incides)
        rec = self.decode(zhat)
        b, _, h, w = rec.shape
        control_feat = F.interpolate(zhat, scale_factor=(w // zhat.shape[-1]) // 8)
        inp_txt, inp_vec = redux_call(self.redux,
            [torchvision.transforms.functional.to_pil_image(torch.clamp((rec[i] + 1.0) / 2.0, 0.0, 1.0)) for i in range(b)],
        )
        prompts = ["" for _ in range(b)]
        result = self.xflux_pipeline(
            prompt=prompts,
            neg_prompt=prompts,
            inp_txt=inp_txt,
            inp_vec=inp_vec,
            controlnet_image=control_feat.to(torch.bfloat16),
            width=w,
            height=h,
            guidance=4.0,
            num_steps=25,
            seed=42,
            true_gs=1.0,
            control_weight=1.0,
            timestep_to_start_cfg=5,
        )
        if self.clamp_range is not None:
            result = torch.clamp(result, self.clamp_range[0], self.clamp_range[1])
        return result.to(dtype=self.ae_dtype)


class AutoencodingFluxLoraEngine(AutoencodingEngine):

    def __init__(self, *args, input_key, lora_path, eval_only = False, encoder_config, decoder_config, loss_config, regularizer_config, optimizer_config = None, lr_g_factor = 1, trainable_ae_params = None, ae_optimizer_args = None, trainable_disc_params = None, disc_optimizer_args = None, disc_start_iter = 0, diff_boost_factor = 3, ckpt_engine = None, ckpt_path = None, additional_decode_keys = None, clamp_range=None, **kwargs):
        super().__init__(*args, input_key=input_key, eval_only=eval_only, encoder_config=encoder_config, decoder_config=decoder_config, loss_config=loss_config, regularizer_config=regularizer_config, optimizer_config=optimizer_config, lr_g_factor=lr_g_factor, trainable_ae_params=trainable_ae_params, ae_optimizer_args=ae_optimizer_args, trainable_disc_params=trainable_disc_params, disc_optimizer_args=disc_optimizer_args, disc_start_iter=disc_start_iter, diff_boost_factor=diff_boost_factor, ckpt_engine=ckpt_engine, ckpt_path=ckpt_path, additional_decode_keys=additional_decode_keys, clamp_range=clamp_range, **kwargs)
        self.lora_path = lora_path
        self.control_channels = encoder_config.params.z_channels

    def load_flux_pipeline(
            self, 
        ):
        device = next(self.parameters()).device
        self.ae_dtype = next(self.parameters()).dtype
        dtype = torch.bfloat16
        self.xflux_pipeline = XFluxPipelineClean("flux-dev", device, dtype)
        flux_dit = load_flow_model_control(self.xflux_pipeline.model_type, self.control_channels, device=device)

        lora_attn_procs = {}
        double_blocks_idx = list(range(19))
        single_blocks_idx = list(range(38))
        for name, attn_processor in flux_dit.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            if name.startswith("double_blocks") and layer_index in double_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                dim=3072, rank=128
                )
            elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                dim=3072, rank=128
                )
            else:
                lora_attn_procs[name] = attn_processor
        flux_dit.set_attn_processor(lora_attn_procs)

        flux_dit.load_state_dict(torch.load(self.lora_path, map_location='cpu'), strict=False)
        flux_dit = flux_dit.to(device=device, dtype=dtype)
        vae = load_ae(self.xflux_pipeline.model_type, device=device)
        self.redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=dtype).to(device)
        self.xflux_pipeline.model = flux_dit
        self.xflux_pipeline.ae = vae
        self.xflux_pipeline.controlnet_loaded = True

    def dequant(self, incides):
        zhat = self.regularization.dequant(incides)
        rec = self.decode(zhat)
        b, _, h, w = rec.shape
        control_feat = F.interpolate(zhat, scale_factor=(w // zhat.shape[-1]) // 8)
        inp_txt, inp_vec = redux_call(self.redux,
            [torchvision.transforms.functional.to_pil_image(torch.clamp((rec[i] + 1.0) / 2.0, 0.0, 1.0)) for i in range(b)],
        )
        prompts = ["" for _ in range(b)]
        result = self.xflux_pipeline.call_plora(
            prompt=prompts,
            neg_prompt=prompts,
            inp_txt=inp_txt,
            inp_vec=inp_vec,
            controlnet_image=control_feat.to(torch.bfloat16),
            width=w,
            height=h,
            guidance=4.0,
            num_steps=25,
            seed=42,
            true_gs=1.0,
            control_weight=1.0,
            timestep_to_start_cfg=5,
        )
        if self.clamp_range is not None:
            result = torch.clamp(result, self.clamp_range[0], self.clamp_range[1])
        return result.to(dtype=self.ae_dtype)

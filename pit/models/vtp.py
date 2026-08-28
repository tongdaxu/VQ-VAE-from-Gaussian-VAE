import torch
import torch.nn as nn
import os
import json
import numpy as np

from huggingface_hub import snapshot_download
from safetensors.torch import load

from pit.modules.vtp_module.vtp_hf import VTPConfig, VTPModel

class VTPVAE(nn.Module):
    def __init__(self, ckpt_folder):
        super().__init__()
        if not os.path.exists(ckpt_folder):
            ckpt_folder = snapshot_download(
                repo_id=ckpt_folder,
            )
        json_path = os.path.join(ckpt_folder, "config.json")
        ckpt_path = os.path.join(ckpt_folder, "model.safetensors")

        with open(json_path) as f:
            vtp_cfg_kwargs = json.load(f)

        vtp_cfg = VTPConfig(**vtp_cfg_kwargs)
        self.vtp = VTPModel(vtp_cfg)

        with open(ckpt_path, "rb") as f:
            ckpt_data = f.read()

        unexp, miss = self.vtp.load_state_dict(load(ckpt_data), strict=False)
        print("unexpected keys", unexp)
        print("missing keys", miss)

    def encode(self, x):
        return self.vtp.get_reconstruction_latents(x)

    def decode(self, z):
        return self.vtp.get_latents_decoded_images(z)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return x, {"xhat": xhat}

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config
    from ifid.fid.psnr import get_psnr
    from PIL import Image

    def read_image_tensor(path):
        """
        Read image and return tensor of shape (1,3,256,256) in range [-1,1]
        """
        img = Image.open(path).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.array(img)).float() / 255.0  # [0,1]
        x = x.permute(2, 0, 1)                               # (3,H,W)
        x = x * 2 - 1                                         # [-1,1]
        x = x.unsqueeze(0)                                    # (1,3,256,256)
        return x

    configs = OmegaConf.load("/video_ssd/kongzishang/xutongda/git/IFID/configs/VTPB.yaml")
    vae = instantiate_from_config(configs).cuda()
    x = read_image_tensor("/video_ssd/kongzishang/xutongda/git/IFID/samples/ifid-rae_extra/src_dist/ILSVRC2012_val_00000236.JPEG").cuda()
    z = vae.encode(x)
    xhat = vae.decode(z)
    psnr = get_psnr(x, xhat, zero_mean=True)
    print(z.shape, psnr)

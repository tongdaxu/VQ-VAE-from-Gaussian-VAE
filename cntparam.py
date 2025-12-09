import torch

sd = torch.load("./models_256/model.ckpt")
param_cnt = 0
for key in sd["state_dict"].keys():
    if "encoder" in key or "decoder" in key:
        param_cnt += torch.numel(sd["state_dict"][key])
print(param_cnt / 1e6)
'''
param_cnt = 0

import torch
from safetensors import safe_open
with safe_open("/workspace/cogview_dev/xutd/hub_cache/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/vae/diffusion_pytorch_model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        if "encoder" in key or "decoder" in key:
            param_cnt += torch.numel(tensor)
print(param_cnt / 1e6)
'''
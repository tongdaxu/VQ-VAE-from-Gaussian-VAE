import torch
from safetensors import safe_open
from safetensors.torch import save_file
import sys
sys.path.append('/workspace/cogview_dev/xutd/xu/models/hub/HunyuanImage-3/')
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
with safe_open("/workspace/cogview_dev/xutd/xu/models/hub/HunyuanImage-3/model-0031-of-0032.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
        if "vae." in key:
            tensors[key[4:]] = f.get_tensor(key)
with safe_open("/workspace/cogview_dev/xutd/xu/models/hub/HunyuanImage-3/model-0032-of-0032.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
        if "vae." in key:
            tensors[key[4:]] = f.get_tensor(key)

miss, unexp = model.load_state_dict(tensors, strict=False)

input = torch.randn([8,3,256,256])
qzx = model.encode(input, return_dict=False)[0]
z = qzx.sample()[:,:,0]
output = model.decode(z[:,:,None], return_dict=False)[0][:,:,0]
print(output.shape)
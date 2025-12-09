import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

model = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae")

image = torch.randn([8,3,256,256])
qzx = model.encode(image[:,:,None],return_dict=False)
z = qzx[0].sample()[:,:,0]
image_out = model.decode(z[:,:,None],return_dict=False)
print(image_out[0][:,:,0].shape)
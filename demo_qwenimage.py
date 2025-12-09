import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

model = AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")

image = torch.randn([8,3,256,256])
qzx = model.encode(image[:,:,None],return_dict=False)
z = qzx[0].sample()[:,:,0]
image_out = model.decode(z[:,:,None],return_dict=False)
print(image_out[0][:,:,0].shape)
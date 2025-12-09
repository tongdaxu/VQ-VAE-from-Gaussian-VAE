from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from pit.util import instantiate_from_config
import torch

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

img = transform(Image.open("demo.png")).unsqueeze(0).cuda()
config = OmegaConf.load("./configs/sd3unet_gq_0.25.yaml")
vae = instantiate_from_config(config.model)
vae.load_state_dict(
    torch.load("models_256/sd3unet_gq_0.25.ckpt",
        map_location=torch.device('cpu'))["state_dict"],strict=False
    )
vae = vae.eval().cuda()

vae.eval()

z = vae.encode(img, return_reg_log=True)[1]["zhat_noquant"] # Gaussian VAE latents
img_hat = vae.decode(z)
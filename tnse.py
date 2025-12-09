import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from openTSNE import TSNE
import tnse_utils as utils
import numpy as np
import torch
from omegaconf import OmegaConf

from pit.util import instantiate_from_config
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from PIL import Image
from glob import glob
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pit.util import instantiate_from_config
from functools import partial
import torchvision.transforms.v2 as transforms

class SimpleDataset(VisionDataset):
    def __init__(self, root: str, image_size=256):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if root.endswith(".txt"):
            with open(root) as f:
                lines = f.readlines()
            self.fpaths = [line.strip("\n") for line in lines]
        else:
            self.fpaths = sorted(glob(root + "/**/*.JPEG", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.jpg", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.png", recursive=True))

        self.fpaths = self.fpaths[:100]
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "img": img,
            "fpath": fpath,
        }


class ComplexDataset(VisionDataset):
    def __init__(self,):
        super().__init__()
        self.names = [
            "kit fox",
            "grey whale",
            "panda",
            "Egyptian cat",
            "sea lion",
        ]

        self.datasets = [
            SimpleDataset("/workspace/cogview_dev/xutd/xu/datasets/ILSVRC/Data/CLS-LOC/train/n02119789"),
            SimpleDataset("/workspace/cogview_dev/xutd/xu/datasets/ILSVRC/Data/CLS-LOC/train/n02066245"),
            SimpleDataset("/workspace/cogview_dev/xutd/xu/datasets/ILSVRC/Data/CLS-LOC/train/n02509815"),
            SimpleDataset("/workspace/cogview_dev/xutd/xu/datasets/ILSVRC/Data/CLS-LOC/train/n02124075"),
            SimpleDataset("/workspace/cogview_dev/xutd/xu/datasets/ILSVRC/Data/CLS-LOC/train/n02077923"),
        ]


    def __len__(self):
        return 500

    def __getitem__(self, index: int):
        dic = self.datasets[index%5].__getitem__(index//5)
        dic["label"] = self.names[index%5]
        return dic


image_dataset = ComplexDataset()

image_dataloader = DataLoader(
    image_dataset,
    8,
    shuffle=False,
    num_workers=32,
    drop_last=True,
)

config = OmegaConf.load("/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/configs/sd3unet_gq_0.25.yaml")
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load("/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/models_256/sd3unet_gq_0.25.ckpt",map_location=torch.device('cpu'))["state_dict"], strict=False)
model = model.eval().cuda()

latents = []
for di, batch in tqdm(enumerate(image_dataloader)):
    if di >= 100:
        break
    img = batch["img"].cuda()
    zhat, info = model.encode(img, return_reg_log=True)
    latents.append(zhat)

latents = torch.cat(latents, dim=0)
latents = latents.reshape(latents.shape[0], -1)
train_data = latents.cpu().numpy()

tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
embedding_train = tsne.fit(train_data)
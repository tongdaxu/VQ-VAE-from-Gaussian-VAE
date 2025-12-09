import natsort
from torchvision.datasets import VisionDataset
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
import torch 
from pit.util import instantiate_from_config
from omegaconf import OmegaConf
import torchvision

class ImageDataset(VisionDataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=["jpg"],
        phase="train",
    ):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f"{str(folder)} must be a folder containing images"
        self.folder = folder

        self.image_size = image_size

        self.paths = natsort.natsorted([p for ext in exts for p in folder.glob(f"**/*.{ext}")])
        print(f"{len(self.paths)} training samples found at {folder}")

        if phase == "train":
            self.transform = T.Compose(
                [
                    T.Resize(image_size),
                    T.RandomHorizontalFlip(),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize(image_size),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        return self.transform(img), "", str(path)


if __name__ == "__main__":
    dataset_val = ImageDataset("/workspace/cogview_dev/zyh/behance_eval_set_v1.2/images/", 256, exts=["jpg"], phase="test")
    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8)

    config = OmegaConf.load("/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/configs/sd3unet_gq_fml_fluxlora.yaml")
    model = instantiate_from_config(config.model)
    model = model.eval().cuda()
    model.load_flux_pipeline()

    for bi, batch in enumerate(val_dataloader):
        img, _, _ = batch
        img = img.cuda()
        with torch.no_grad():
            zhat, indices = model.quant(img)
            rec = model.dequant(indices)

        torchvision.utils.save_image(img[0], "src.png", normalize=True, value_range=(-1,1))
        torchvision.utils.save_image(rec[0], "rec.png", normalize=True, value_range=(-1,1))

        assert(0)
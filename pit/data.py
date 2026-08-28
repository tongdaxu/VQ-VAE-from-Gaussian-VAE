from torchvision.datasets import VisionDataset
from PIL import Image
from glob import glob
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pit.util import instantiate_from_config
from functools import partial
import torchvision.transforms.v2 as transforms


class ImageDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        num_workers=None,
        shuffle_test_loader=False,
        shuffle_val_dataloader=False,
        prefetch_factor=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader)
        self.prefetch_factor = prefetch_factor

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )

    def _train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def _val_dataloader(self, shuffle=False):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
        )

    def _test_dataloader(self, shuffle=False):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )


class SimpleDataset(VisionDataset):
    def __init__(self, root: str, image_size):
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

        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        try:
            fpath = self.fpaths[index]
            img = Image.open(fpath).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {fpath}: {e}")
            return self.__getitem__((index + 1) % len(self.fpaths))

        return {
            "img": img,
            "fpath": fpath,
        }

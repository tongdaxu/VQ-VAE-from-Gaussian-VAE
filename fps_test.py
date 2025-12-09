import argparse
import os
import torch.distributed as dist
from torchvision.utils import save_image

# local_rank = int(os.environ["LOCAL_RANK"])
# os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from omegaconf import OmegaConf

from pit.util import instantiate_from_config
import torch.nn.functional as F

from pit.evaluations.fid.fid_score import calculate_frechet_distance
from pit.evaluations.lpips import get_lpips
from pit.evaluations.psnr import get_psnr
from pit.evaluations.ssim import get_ssim_and_msssim
from pit.evaluations.fid.inception import InceptionV3
from pit.data import SimpleDataset

def print_dict(dict_stat):
    for key in dict_stat.keys():
        print("{0} -- mean: {1:.4f}".format(key, np.mean(dict_stat[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        choices=["nccl", "gloo"],
        type=str,
        help="distributed backend",
    )
    parser.add_argument(
        "--base",
        default="",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="",
        type=str,
    )
    parser.add_argument(
        "--img_size",
        default=265,
        type=int,
    )
    parser.add_argument(
        "--bs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--save",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
    )
    args = parser.parse_args()

    dist.init_process_group(
        backend=args.dist_backend,
        init_method="env://",
    )

    world_size = dist.get_world_size()

    BS = args.bs

    image_dataset = SimpleDataset(args.dataset, image_size=args.img_size)

    image_sampler = torch.utils.data.distributed.DistributedSampler(
        image_dataset, shuffle=False
    )
    image_dataloader = DataLoader(
        image_dataset,
        BS,
        shuffle=False,
        num_workers=8,
        sampler=image_sampler,
        drop_last=True,
    )

    config = OmegaConf.load(args.base)

    model = instantiate_from_config(config.model)
    model = model.eval().cuda()
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx], normalize_input=False).cuda()
    inception_v3.eval()

    all_pred_x = [[] for _ in range(world_size)]
    all_pred_xr = [[] for _ in range(world_size)]
    all_psnr = [[] for _ in range(world_size)]
    all_ssim = [[] for _ in range(world_size)]
    all_msssim = [[] for _ in range(world_size)]
    all_lpips = [[] for _ in range(world_size)]
    all_hist = torch.zeros([65536])

    if args.save:
        src_dir = os.path.join(args.save_dir, "src")
        rec_dir = os.path.join(args.save_dir, "rec")
        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(rec_dir, exist_ok=True)

    total_num = 0

    def cal_ent(hist):
        usage = torch.sum((hist == 0).to(dtype=torch.float32)) / hist.shape[0]
        hist = hist / torch.sum(hist)
        ent = - torch.sum(hist * torch.log2(hist + 1e-5))
        return 1 - usage, ent

    with torch.no_grad():

        for ii, (batch) in tqdm(enumerate(image_dataloader)):
            fpaths = batch["fpath"]
            img = batch["img"]
            img = img.cuda()
            zhat, info = model.encode(img, return_reg_log=True)

            # rec = model.decode(zhat)


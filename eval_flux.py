import argparse
import os
import torch.distributed as dist
from torchvision.utils import save_image

local_rank = int(os.environ["LOCAL_RANK"])

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
    model = model.eval().cuda(local_rank)
    model.load_flux_pipeline()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx], normalize_input=False).cuda(local_rank)
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
            img = img.cuda(local_rank)
            zhat, indices = model.quant(img)
            rec = model.dequant(indices)
            # eval metrics ...
            # PSNR
            if args.save:
                for b in range(len(fpaths)):
                    fname = fpaths[b].split('/')[-1] + ".png"
                    src_path = os.path.join(src_dir, fname)
                    save_image(img[b], src_path, normalize=True, value_range=(-1, 1))
                    rec_path = os.path.join(rec_dir, fname)
                    save_image(rec[b], rec_path, normalize=True, value_range=(-1, 1))
                
            pred_psnr = get_psnr(img, rec, zero_mean=True)
            gathered_psnr = [torch.zeros_like(pred_psnr) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_psnr, pred_psnr)
            for j in range(world_size):
                all_psnr[j].append(gathered_psnr[j].detach().cpu())
            # SSIM
            pred_ssim, pred_msssim = get_ssim_and_msssim(img, rec, zero_mean=True)
            gathered_ssim = [torch.zeros_like(pred_ssim) for _ in range(world_size)]
            gathered_msssim = [torch.zeros_like(pred_msssim) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_ssim, pred_ssim)
            torch.distributed.all_gather(gathered_msssim, pred_msssim)
            for j in range(world_size):
                all_ssim[j].append(gathered_ssim[j].detach().cpu())
                all_msssim[j].append(gathered_msssim[j].detach().cpu())
            # LPIPS (AlexNet)
            pred_lpips = get_lpips(img, rec, zero_mean=True, network_type="alex")
            gathered_lpips = [torch.zeros_like(pred_lpips) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_lpips, pred_lpips)
            for j in range(world_size):
                all_lpips[j].append(gathered_lpips[j].detach().cpu())

            # fid
            pred_x = inception_v3(img)[0]
            if pred_x.size(2) != 1 or pred_x.size(3) != 1:
                pred_x = F.adaptive_avg_pool2d(pred_x, (1, 1))
            pred_x = pred_x.squeeze(3).squeeze(2)
            gathered_pred_x = [torch.zeros_like(pred_x) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_pred_x, pred_x)
            for j in range(world_size):
                all_pred_x[j].append(gathered_pred_x[j].detach().cpu())

            pred_xr = inception_v3(rec)[0]
            if pred_xr.size(2) != 1 or pred_xr.size(3) != 1:
                pred_xr = F.adaptive_avg_pool2d(pred_xr, (1, 1))
            pred_xr = pred_xr.squeeze(3).squeeze(2)
            gathered_pred_xr = [torch.zeros_like(pred_xr) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_pred_xr, pred_xr)
            for j in range(world_size):
                all_pred_xr[j].append(gathered_pred_xr[j].detach().cpu())

            total_num += world_size * img.shape[0]

            """
            for jj in range(BS):
                iname = inames[jj].split("/")[-1].split(".")[0] + ".png"
                torchvision.utils.save_image(
                    img[jj:jj+1], os.path.join(subdirs[0], iname), normalize=True, value_range=(-1,1)
                )
                torchvision.utils.save_image(
                    rec2[jj:jj+1], os.path.join(subdirs[1], iname), normalize=True, value_range=(-1,1)
                )
            """

        if local_rank == 0:
            for j in range(world_size):
                all_psnr[j] = torch.cat(all_psnr[j], dim=0).numpy()
            all_psnr_reorg = []
            for j in range(total_num):
                all_psnr_reorg.append(all_psnr[j % world_size][j // world_size])
            all_psnr = np.vstack(all_psnr_reorg)
            print(f"PSNR: {np.mean(all_psnr):.4f} (±{np.std(all_psnr):.4f})")
            # SSIM
            for j in range(world_size):
                all_ssim[j] = torch.cat(all_ssim[j], dim=0).numpy()
                all_msssim[j] = torch.cat(all_msssim[j], dim=0).numpy()
            all_ssim_reorg = []
            all_msssim_reorg = []
            for j in range(total_num):
                all_ssim_reorg.append(all_ssim[j % world_size][j // world_size])
                all_msssim_reorg.append(all_msssim[j % world_size][j // world_size])
            all_ssim = np.vstack(all_ssim_reorg)
            all_msssim = np.vstack(all_msssim_reorg)
            print(f"SSIM: {np.mean(all_ssim):.4f} (±{np.std(all_ssim):.4f})")
            print(f"MS-SSIM: {np.mean(all_msssim):.4f} (±{np.std(all_msssim):.4f})")
            # LPIPS
            for j in range(world_size):
                all_lpips[j] = torch.cat(all_lpips[j], dim=0).numpy()
            all_lpips_reorg = []
            for j in range(total_num):
                all_lpips_reorg.append(all_lpips[j % world_size][j // world_size])
            all_lpips = np.vstack(all_lpips_reorg)
            print(
                f"LPIPS (AlexNet): {np.mean(all_lpips):.4f} (±{np.std(all_lpips):.4f})"
            )

            for j in range(world_size):
                all_pred_x[j] = torch.cat(all_pred_x[j], dim=0).numpy()
            all_pred_x_reorg = []
            for j in range(total_num):
                all_pred_x_reorg.append(all_pred_x[j % world_size][j // world_size])
            all_pred_x = np.vstack(all_pred_x_reorg)
            all_pred_x = all_pred_x
            m2, s2 = np.mean(all_pred_x, axis=0), np.cov(all_pred_x, rowvar=False)

            for j in range(world_size):
                all_pred_xr[j] = torch.cat(all_pred_xr[j], dim=0).numpy()
            all_pred_xr_reorg = []
            for j in range(total_num):
                all_pred_xr_reorg.append(all_pred_xr[j % world_size][j // world_size])
            all_pred_xr = np.vstack(all_pred_xr_reorg)
            all_pred_xr = all_pred_xr
            m1, s1 = np.mean(all_pred_xr, axis=0), np.cov(all_pred_xr, rowvar=False)

            fid_score = calculate_frechet_distance(m1, s1, m2, s2)
            print(f"FID: {fid_score:.4f}")

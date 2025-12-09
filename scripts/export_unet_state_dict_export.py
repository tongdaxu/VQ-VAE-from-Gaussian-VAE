import torch

if __name__ == "__main__":
    model = torch.load("/workspace/cogview_dev/xutd/xu/generative-models/logs/2025-10-27T02-21-06_sd3.5_c16_kl_0.10_wds_vavae_d16c64_4n_0.25_0.1_vres_full/checkpoints/trainstep_checkpoints/epoch=000000-step=000200000.ckpt")
    stats = torch.load("/workspace/cogview_dev/xutd/xu/LightningDiT/datacache/sd3.5_c16_kl_0.10_wds_vavae_d16c64_4n_0.25_0.1_vres_full_export/imagenet_train_256/latents_stats.pt")
    keys = list(model.keys())
    print(keys)
    model["state_dict"]["latent_mean"] = stats["mean"]
    model["state_dict"]["latent_std"] = stats["std"]

    torch.save(model, "../models_256/sd3.5_c16_kl_0.10_wds_vavae_d16c64_4n_0.25_0.1_vres_full_export.ckpt")

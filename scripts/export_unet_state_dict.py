import torch

if __name__ == "__main__":
    model = torch.load("/workspace/cogview_dev/xutd/xu/generative-models/logs/2025-10-14T04-10-46_sd3.5_c16_kl_0.10_wds_vavae_d8c16_4n_vres/checkpoints/trainstep_checkpoints/epoch=000000-step=000100000.ckpt")
    keys = list(model.keys())
    for key in keys:
        if key == "state_dict":
            continue
        del model[key]
    torch.save(model, "../models_256/sd3.5_c16_kl_0.10_wds_vavae_d8c16_4n_vres.ckpt")

import torch

if __name__ == "__main__":
    model = torch.load("/workspace/cogview_dev/xutd/xu/bsq-vit/logs/imagenet_256x256_ta_t_16_g_16_stylegan_f8_fp16_lr_2e-7/checkpoint.pt")
    model = model["state_dict"]
    new_model = {
        "state_dict": {}
    }
    keys = list(model.keys())
    for key in keys:
        new_key = key.replace('module.', '')
        if "post_quant_embed" in new_key:
            new_key = "decoder." + new_key
        elif "quant_embed" in new_key:
            new_key = "encoder." + new_key
        new_model["state_dict"][new_key] = model[key]
    torch.save(new_model, "../models_256/bsqvit_gq_0.25.ckpt")

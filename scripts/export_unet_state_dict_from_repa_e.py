import torch

if __name__ == "__main__":
    model = torch.load("/workspace/cogview_dev/xutd/xu/REPA-E/exps/sit-xl-dinov2-b-enc8-repae-flow-6r-0.1_ttut_40-imagekl_ch16_225w-0.5-1.5-400k/checkpoints/0400000.pt")
    keys = list(model.keys())
    new_model = {}
    new_model["state_dict"] = model["vae"]
    torch.save(new_model, "../models_256/imagekl_ch16_225w_repae_flow_6r_0.1_ttut_40.ckpt")

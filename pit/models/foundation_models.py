"""
We use the file to instantiate the vision foundation models.
They serves as the auxiliary regularizer for the autoencoder.

by Jingfeng Yao
from HUST-VL
"""

import timm
import torch
import torch.nn as nn

def get_mae_encoder():
    """
    Load the MAE pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch16_224.mae", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def get_dinov2_encoder():
    """
    Load the DINOv2 pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def get_dinov3_encoder(feat_indice):
    model = timm.create_model("hf-hub:timm/vit_large_patch16_dinov3.lvd_1689m", pretrained=True, dynamic_img_size=True, features_only=True, 
                              out_indices=feat_indice)
    model.requires_grad_(False)
    return model

def create_foundation_model(
    type,
):
    assert type in ['mae', 'dinov2', 'dinov3'], f"Unsupported foundation model type: {type}"

    if type == 'mae':
        return get_mae_encoder(), 1024
    elif type == 'dinov2':
        return get_dinov2_encoder(), 1024
    elif type == 'dinov3':
        return get_dinov3_encoder(), 1024 * 3

def renorm_dinov3(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    x_01 = (x + 1.0) / 2.0
    x_norm = (x_01 - mean) / std
    return x_norm

class aux_foundation_model(nn.Module):
    """
    Load the foundation model and forward the input image to get 
    the feature maps.
    """
    def __init__(self, type, use_renorm=False, use_layer=-1):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type)
        self.type = type
        self.feature_dim = feature_dim
        self.use_renorm = use_renorm
        self.use_layer = use_layer

    def forward_mae(self, x):
        b, c, h, w = x.shape
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
    
    def forward_dinov2(self, x):
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        if self.use_layer == -1 or self.use_layer == 23:
            return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
        else:
            _, intermediates = self.model.forward_intermediates(x)
            return intermediates[self.use_layer].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)


    def forward_dinov3(self, x):
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return torch.cat(self.model(x),dim=1)

    def forward(self, x):
        if self.use_renorm:
            x = renorm_dinov3(x)
        with torch.no_grad():
            if self.type == 'mae':
                return self.forward_mae(x)
            elif self.type == 'dinov2':
                return self.forward_dinov2(x)
            elif self.type == 'dinov3':
                return self.forward_dinov3(x)
            else:
                assert(0)

class DINOEncoder(nn.Module):
    def __init__(self, feat_indice, z_channels, down_factor):
        super().__init__()
        self.model = get_dinov3_encoder(feat_indice)
        self.feature_dim = 1024 * len(feat_indice) 
        self.conv_out = nn.Conv2d(self.feature_dim, z_channels, kernel_size=1, bias=False)
        self.down_factor=down_factor

    def forward_dinov3(self, x):
        b, c, h, w = x.shape
        x_in = renorm_dinov3(x)
        outputs = self.model(x_in)
        return torch.cat(outputs,dim=1)

    def forward(self, x):
        with torch.no_grad():
            x = self.forward_dinov3(x)
        return self.conv_out(x)


if __name__ == "__main__":
    model = DINOEncoder([-1], 1024, 8).cuda()
    img = torch.randn([1,3,256,256]).cuda()
    out = model(img)
    print(out.shape)
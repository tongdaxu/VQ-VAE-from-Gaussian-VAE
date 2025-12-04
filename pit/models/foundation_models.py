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

def get_dinov3_encoder():
    model = timm.create_model("hf-hub:timm/vit_large_patch16_dinov3.lvd_1689m", pretrained=True, dynamic_img_size=True, features_only=True)
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

class aux_foundation_model(nn.Module):
    """
    Load the foundation model and forward the input image to get 
    the feature maps.
    """
    def __init__(self, type, down_factor=16):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type)
        self.type = type
        self.feature_dim = feature_dim
        self.down_factor=down_factor

    def forward_mae(self, x):
        b, c, h, w = x.shape
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
    
    def forward_dinov2(self, x):
        x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        b, c, h, w = x.shape
        if self.down_factor==16:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
        elif self.down_factor==8:
            x = nn.functional.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
            return self.model.forward_features(x)[:, 1:].reshape(b, h//8, w//8, -1).permute(0, 3, 1, 2)

    def forward_dinov3(self, x):
        b, c, h, w = x.shape
        if self.down_factor==16:
            return torch.cat(self.model(x),dim=1)
        elif self.down_factor==8:
            x = nn.functional.interpolate(x, size=(h*2, w*2), mode='bilinear', align_corners=False)
            return torch.cat(self.model(x),dim=1)

    def forward(self, x):
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
    def __init__(self, type, z_channels, down_factor):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type)
        self.feature_dim = feature_dim
        self.conv_out = nn.Conv2d(self.feature_dim, z_channels, kernel_size=1, bias=False)
        self.down_factor=down_factor

    def forward_dinov3(self, x):
        b, c, h, w = x.shape
        if self.down_factor==16:
            return torch.cat(self.model(x),dim=1)
        elif self.down_factor==8:
            x = nn.functional.interpolate(x, size=(h*2, w*2), mode='bilinear', align_corners=False)
            return torch.cat(self.model(x),dim=1)

    def forward(self, x):
        with torch.no_grad():
            x = self.forward_dinov3(x)
        return self.conv_out(x)


if __name__ == "__main__":
    model = DINOEncoder("dinov3", 1024, 8).cuda()
    img = torch.randn([1,3,256,256]).cuda()
    out = model(img)
    print(out.shape)
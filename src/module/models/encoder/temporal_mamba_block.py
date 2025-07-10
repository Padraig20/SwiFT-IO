import torch
import torch.nn as nn
from monai.networks.blocks import MLPBlock as Mlp

class TemporalMambaBlock4D(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act_layer="GELU"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer, dropout_rate=drop)

        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except ImportError:
            raise ImportError("Please install mamba-ssm to use TemporalMambaBlock4D")

        self.mamba = Mamba(dim)

    def forward(self, x):
        # x: [B, D, H, W, T, C] → Mamba over T
        B, D, H, W, T, C = x.shape
        x = x.permute(0, 1, 2, 3, 5, 4).reshape(-1, C, T).transpose(1, 2)  # [B*D*H*W, T, C]
        x = self.norm1(x)
        x = self.mamba(x)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.transpose(1, 2).reshape(B, D, H, W, C, T).permute(0, 1, 2, 3, 5, 4)  # [B, D, H, W, T, C]
        return x
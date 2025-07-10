import torch
import torch.nn as nn
from .swin4d_transformer_ver9 import SwinTransformer4D as SwinTransformer4D_Base
from .temporal_mamba_block import TemporalMambaBlock4D

class MambaLayer(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=4.0, drop=0.0, act_layer="GELU"):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalMambaBlock4D(dim=dim, mlp_ratio=mlp_ratio, drop=drop, act_layer=act_layer)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SwinTransformer4D(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.base = SwinTransformer4D_Base(**kwargs)

        # 마지막 layer만 Temporal Mamba로 대체
        last_layer_index = len(self.base.layers) - 1
        last_dim = int(kwargs["embed_dim"] * kwargs.get("c_multiplier", 2) ** last_layer_index)
        last_depth = kwargs["depths"][last_layer_index]
        mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        drop = kwargs.get("drop_rate", 0.0)

        self.base.layers[last_layer_index] = MambaLayer(
            dim=last_dim, depth=last_depth, mlp_ratio=mlp_ratio, drop=drop
        )

    def forward(self, x):
        return self.base(x)

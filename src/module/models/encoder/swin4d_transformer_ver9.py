"""
Our code is based on the following code.
https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html#SwinUNETR
"""

import itertools
from typing import Optional, Sequence, Tuple, Type, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
import torch.nn.functional as F

from monai.networks.blocks import MLPBlock as Mlp

from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from .patchembedding import PatchEmbed

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "window_partition",
    "window_reverse",
    "WindowAttention4D",
    "SwinTransformerBlock4D",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer4D",
]

def get_1d_sincos_pos_embed(embed_dim, seq_len):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    pos_embed = torch.zeros(seq_len, embed_dim)
    pos = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * 
                         -(math.log(10000.0) / embed_dim))
    pos_embed[:, 0::2] = torch.sin(pos.float() * div_term)
    pos_embed[:, 1::2] = torch.cos(pos.float() * div_term)
    return pos_embed 


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Partition tokens into their respective windows

     Args:
        x: input tensor (B, D, H, W, T, C)

        window_size: local window size.


    Returns:
        windows: (B*num_windows, window_size*window_size*window_size*window_size, C)
    """
    x_shape = x.size()

    b, d, h, w, t, c = x_shape
    x = x.view(
        b,
        d // window_size[0],  # number of windows in depth dimension
        window_size[0],  # window size in depth dimension
        h // window_size[1],  # number of windows in height dimension
        window_size[1],  # window size in height dimension
        w // window_size[2],  # number of windows in width dimension
        window_size[2],  # window size in width dimension
        t // window_size[3],  # number of windows in time dimension
        window_size[3],  # window size in time dimension
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9)
        .contiguous()
        .view(-1, window_size[0] * window_size[1] * window_size[2] * window_size[3], c)
    )
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor (B*num_windows, window_size, window_size, C)
        window_size: local window size.
        dims: dimension values.

    Returns:
        x: (B, D, H, W, T, C)
    """

    b, d, h, w, t = dims
    x = windows.view(
        b,
        torch.div(d, window_size[0], rounding_mode="floor"),
        torch.div(h, window_size[1], rounding_mode="floor"),
        torch.div(w, window_size[2], rounding_mode="floor"),
        torch.div(t, window_size[3], rounding_mode="floor"),
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        -1,
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous().view(b, d, h, w, t, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention4D(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_MuTransfer: bool = False,
        use_flashattn: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 if use_MuTransfer == False else head_dim ** -1 # 250204 jubin added. 
        self.use_flashattn = use_flashattn # 250212 jubin added.
        self.dropout_p = attn_drop
        mesh_args = torch.meshgrid.__kwdefaults__

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_flashattn:
            if mask is not None:
                nw = mask.shape[0]
                mask = torch.cat([mask.to(q.dtype)]*(b_//nw)).unsqueeze(1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p, scale=self.scale)  
            x = attn.transpose(1, 2).reshape(b_, n, c)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock4D(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
        use_MuTransfer: bool = False,
        use_flashattn: bool = False,
        use_post_norm: bool = False
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.use_MuTransfer = use_MuTransfer
        self.use_post_norm = use_post_norm

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention4D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_MuTransfer=use_MuTransfer,
            use_flashattn=use_flashattn
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        if not self.use_post_norm:
            x = self.norm1(x)

        b, d, h, w, t, c = x.shape
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        pad_d0 = pad_h0 = pad_w0 = pad_t0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_h1 = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_w1 = (window_size[2] - w % window_size[2]) % window_size[2]
        pad_t1 = (window_size[3] - t % window_size[3]) % window_size[3]
        x = F.pad(x, (0, 0, pad_t0, pad_t1, pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1))  # last tuple first in
        _, dp, hp, wp, tp, _ = x.shape
        dims = [b, dp, hp, wp, tp]
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2], -shift_size[3]), dims=(1, 2, 3, 4)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2], shift_size[3]), dims=(1, 2, 3, 4)
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_h1 > 0 or pad_w1 > 0 or pad_t1 > 0:
            x = x[:, :d, :h, :w, :t, :].contiguous()
        if self.use_post_norm:
            x = self.norm1(x)

        return x

    def forward_part2(self, x):
        if self.use_post_norm:
            x = self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
        c_multiplier: int = 2,
        use_post_norm: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        self.use_post_norm = use_post_norm

        # Skip dimension reduction on the temporal dimension

        self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(c_multiplier * dim) if use_post_norm else norm_layer(8 * dim) 


    def forward(self, x):
        x_shape = x.size()
        b, d, h, w, t, c = x_shape
        x = torch.cat(
            [x[:, i::2, j::2, k::2, :, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
            -1,
        )

        if self.use_post_norm:
            x = self.norm(self.reduction(x))
        else:
            x = self.reduction(self.norm(x))

        return x

MERGING_MODE = {"mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    d, h, w, t = dims
    img_mask = torch.zeros((1, d, h, w, t, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                for t in slice(-window_size[3]), slice(-window_size[3], -shift_size[3]), slice(-shift_size[3], None):
                    img_mask[:, d, h, w, t, :] = cnt
                    cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        use_MuTransfer: bool = False,
        use_flashattn: bool = False,
        use_post_norm: bool = False
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            use_MuTransfer: use mu transfer.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_MuTransfer = use_MuTransfer
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    use_MuTransfer=use_MuTransfer,
                    use_flashattn=use_flashattn,
                    use_post_norm=use_post_norm
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim,
                norm_layer=norm_layer,
                spatial_dims=len(self.window_size),
                c_multiplier=c_multiplier,
                use_post_norm=use_post_norm
            )

    def forward(self, x):
        b, c, d, h, w, t = x.size()
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        tp = int(np.ceil(t / window_size[3])) * window_size[3]
        attn_mask = compute_mask([dp, hp, wp, tp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, t, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

# Basic layer for full attention,
# the only difference is that there is no window shifting
class BasicLayer_FullAttention(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        use_MuTransfer: bool = False,
        use_flashattn: bool = False,
        use_post_norm: bool = False
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_MuTransfer = use_MuTransfer
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    use_MuTransfer=use_MuTransfer,
                    use_flashattn=use_flashattn,
                    use_post_norm=use_post_norm
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim,
                norm_layer=norm_layer,
                spatial_dims=len(self.window_size),
                c_multiplier=c_multiplier,
                use_post_norm=use_post_norm
            )

    def forward(self, x):
        b, c, d, h, w, t = x.size()
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        tp = int(np.ceil(t / window_size[3])) * window_size[3]
        attn_mask = None
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, t, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)
            
class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self, dim: int, patch_num: tuple, emb_type='spatio_temporal'
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            patch_num: total number of patches per time frame
            time_num: total number of time frames
        """

        super().__init__()
        self.dim = dim
        self.patch_num = patch_num
        self.emb_type = emb_type
        d, h, w, t = patch_num
        if self.emb_type == 'spatio_temporal':
            self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w, 1))
            self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, t))
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.time_embed, std=0.02)
        elif self.emb_type == 'spatial':
            self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w, 1))
            trunc_normal_(self.pos_embed, std=0.02)
        elif self.emb_type == 'temporal':
            self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, t), requires_grad=False)
            time_embed = get_1d_sincos_pos_embed(embed_dim=dim, seq_len=t)  # t C
            self.time_embed.data.copy_(time_embed.reshape(1, dim, 1, 1, 1, t))
        

    def forward(self, x):
        b, c, d, h, w, t = x.shape
        if self.emb_type == 'spatio_temporal':
            x = x + self.pos_embed
            # only add time_embed up to the time frame of the input in case the input size changes
            x = x + self.time_embed[:, :, :, :, :, :t]
        elif self.emb_type == 'spatial':
            x = x + self.pos_embed
        elif self.emb_type == 'temporal':       
            x = x + self.time_embed[:, :, :, :, :, :t]     

        return x


class SwinTransformer4D(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        img_size: Tuple,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        first_window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 4,
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = False,
        downsample="mergingv2",
        num_classes=2,
        to_float: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).


            c_multiplier: multiplier for the feature length after patch merging
        """

        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.first_window_size = first_window_size
        self.patch_size = patch_size
        self.to_float = to_float
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            flatten=True, # kimbo change: changed from False to True
            spatial_dims=spatial_dims,
        )
        grid_size = self.patch_embed.grid_size
        self.grid_size = grid_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.use_MuTransfer = kwargs.get('use_MuTransfer', False)
        self.use_flashattn = kwargs.get('use_flashattn', False)
        self.use_post_norm = kwargs.get('use_post_norm', False)
        if self.use_flashattn:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                print("Flash Attention is not supported on this device. Using normal attention instead.")
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        #patch_num = int((img_size[0]/patch_size[0]) * (img_size[1]/patch_size[1]) * (img_size[2]/patch_size[2]))
        #time_num = int(img_size[3]/patch_size[3])
        patch_num =  ((img_size[0]//patch_size[0]), (img_size[1]//patch_size[1]), (img_size[2]//patch_size[2]), (img_size[3]//patch_size[3]))

        #print img, patch size, patch dim
        print("img_size: ", img_size)
        print("patch_size: ", patch_size)
        print("patch_num: ", patch_num)
        self.pos_embeds = nn.ModuleList()
        pos_embed_dim = embed_dim
        for i in range(self.num_layers):
            self.pos_embeds.append(PositionalEmbedding(pos_embed_dim, patch_num))
            if i < self.num_layers -1:
                pos_embed_dim = pos_embed_dim * c_multiplier
                patch_num = (patch_num[0]//2, patch_num[1]//2, patch_num[2]//2, patch_num[3])
        self.encoder_out_patch_num = patch_num
        # build layer
        self.layers = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
    
        layer = BasicLayer(
            dim=int(embed_dim),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.first_window_size,
            drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            c_multiplier=c_multiplier,
            downsample=down_sample_mod if 0 < self.num_layers - 1 else None,
            use_checkpoint=use_checkpoint,
            use_MuTransfer=self.use_MuTransfer,
            use_flashattn=self.use_flashattn,
            use_post_norm=self.use_post_norm
        )
        self.layers.append(layer)

        # exclude last layer
        for i_layer in range(1, self.num_layers - 1):
            layer = BasicLayer(
                dim=int(embed_dim * (c_multiplier**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=down_sample_mod if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                use_MuTransfer=self.use_MuTransfer,
                use_flashattn=self.use_flashattn,
                use_post_norm=self.use_post_norm
            )
            self.layers.append(layer)

        if not last_layer_full_MSA:
            layer = BasicLayer(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
                use_MuTransfer=self.use_MuTransfer,
                use_flashattn=self.use_flashattn,
                use_post_norm=self.use_post_norm
            )
            self.layers.append(layer)

        else:
            #################Full MSA for last layer#####################

            self.last_window_size = (
                self.grid_size[0] // int(2 ** (self.num_layers - 1)),
                self.grid_size[1] // int(2 ** (self.num_layers - 1)),
                self.grid_size[2] // int(2 ** (self.num_layers - 1)),
                self.window_size[3],
            )

            layer = BasicLayer_FullAttention(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                # change the window size to the entire grid size
                window_size=self.last_window_size,
                drop_path=dpr[sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
                use_MuTransfer=self.use_MuTransfer,
                use_flashattn=self.use_flashattn,
                use_post_norm=self.use_post_norm
            )
            self.layers.append(layer)

            #############################################################

        encoder_out_dim = int(embed_dim * c_multiplier ** (self.num_layers - 1))

        self.norm = norm_layer(encoder_out_dim)
        #self.avgpool = nn.AdaptiveAvgPool1d(1)  #
        #self.head = nn.Linear(self.num_features, 1) if num_classes == 2 else num_classes

        # weight initialize 
        self.init_weights()


    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)
        if self.use_post_norm:
            for bly in self.layers:
                bly._init_respostnorm()


    def mup_init_weights(self):
        print("MuP Used")
        import mup
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                mup.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)


    def set_MuReadout_layer(self):
        # 250402 jubin added
        # unlike simmim model, we are not using the last layer for readout since output_head is set outside the model
        pass
        

    def forward(self, x: torch.Tensor):
        if self.to_float:
            # converting tensor to float
            x = x.float()

       # patch embedding
        x = self.patch_embed(x)     # B C D H W T 
        B, C, D, H, W, T = x.shape
        x = self.pos_drop(x)  # B C D H W T 

        # swin blocks 
        for i in range(self.num_layers):
            x = self.pos_embeds[i](x)
            x = self.layers[i](x.contiguous())  # B C D H W T 
        
        # last norm
        b, c, d, h, w, t = x.shape 
        x = x.flatten(2).permute(0, 2, 1)  # B L C 
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(b, c, d, h, w, t)

        # for decoder
        x = x.flatten(start_dim=2) #.transpose(1, 2) # B L C
        
        return x 

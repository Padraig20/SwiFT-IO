"""
Our code is based on the following code.
https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html#SwinUNETR
"""
import itertools
import math
from typing import Any, Optional, Tuple, List, Type, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from .patchembedding import PatchEmbed

rearrange, _ = optional_import("einops", name="rearrange")


def change_st_to_ts_dimension(
    input_shape=[96, 96, 96, 36]
    ): 
    t = input_shape[-1]
    s = input_shape[:-1]
    if isinstance(t, int) or isinstance(t, float): 
        t = [t]
    if isinstance(s, int) or isinstance(s, float): 
        s = [s]
    elif isinstance(s, tuple): 
        s = list(s)

    return t + s



class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        input_resolution,
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
        self.input_resolution = input_resolution
        self.dim = dim
        self.use_post_norm = use_post_norm

        # Skip dimension reduction on the temporal dimension

        self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(c_multiplier * dim) if use_post_norm else norm_layer(8 * dim) 


    def forward(self, x):
        B, _, C = x.size()
        t_in, d_in, h_in, w_in = self.input_resolution
        x = x.reshape(B, t_in, d_in, h_in, w_in, C)
        x = torch.cat(
            [x[:, :, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
            -1,
        )

        x = self.reduction(self.norm(x))
        x = x.permute(0, 5, 1, 2, 3, 4).contiguous() # B C_out, t_out, d_out, h_out, w_out
        x = x.flatten(2).transpose(1, 2)  # B C_out L  -> B L C_out 
        return x


class PixelShuffle4D(nn.Module):
    '''
    reference: http://www.multisilicon.com/blog/a25332339.html
    Onlt shuffle spatial dims. Not time dims
    '''
    def __init__(self,spatial_scale, temporal_scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.spatial_scale = spatial_scale 
        self.temporal_scale = temporal_scale

    def forward(self, input: torch.Tensor):
        batch_size, in_depth, in_height, in_width, in_time, channels = input.size()
        #nOut = channels // self.scale ** 3    # original version
        nOut = int(channels // ((self.spatial_scale ** 3) * self.temporal_scale))     # torchscript version
        """
        out_depth = in_depth * self.scale 
        out_height = in_height * self.scale 
        out_width = in_width * self.scale 
        
        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width, time)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous() 
        return output.view(batch_size, nOut, out_depth, out_height, out_width) 
        """
        output = rearrange(input, 'B d h w t (C s1 s2 s3 s4) -> B (s1 d) (s2 h) (s3 w) (s4 t) C', 
                           B=batch_size, d=in_depth, h=in_height, w=in_width, t=in_time, C=nOut,
                           s1=self.spatial_scale, s2=self.spatial_scale, s3=self.spatial_scale, s4=self.temporal_scale)
        
        return output


MERGING_MODE = {"mergingv2": PatchMergingV2}
EXPANDING_MODE = {"pixelshuffle": PixelShuffle4D}

class PatchEmbed(nn.Module):
    """ 4D Image to Patch Embedding
    """

    def __init__(
        self,
        img_size = (20, 96, 96, 96),
        patch_size=(1, 4, 4, 4),
        in_chans=1,
        embed_dim=24,
        norm_layer=None,
        flatten=True,
    ):
        assert len(patch_size) == 4, "you have to give four numbers, each corresponds t, d, h, w"
        #assert patch_size[3] == 1, "temporal axis merging is not implemented yet"

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
            img_size[3] // patch_size[3],
        )
        self.embed_dim = embed_dim
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2] * self.patches_resolution[3]
        self.flatten = flatten

        self.fc = nn.Linear(in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3], out_features=embed_dim)

        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()


    def proj(self, x):
        B, C, T, D, H, W = x.shape
        pT, pD, pH, pW = self.patches_resolution
        sT, sD, sH, sW = self.patch_size

        x = x.view(B, C, pT, sT, pD, sD, pH, sH, pW, sW)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sT * sD * sH * sW * C) # => B, pT, pD, pH, pW, sT, sD, sH, sW, C
        x = self.fc(x)
        x = x.view(B, pT, pD, pH, pW, self.embed_dim).contiguous()
        x = x.permute(0, 5, 1, 2, 3, 4)
        return x


    def forward(self, x):
        B, C, T, D, H, W = x.shape
        assert D == self.img_size[1], f"Input image height ({D}) doesn't match model ({self.img_size[1]})."
        assert H == self.img_size[2], f"Input image width ({H}) doesn't match model ({self.img_size[2]})."
        assert W == self.img_size[3], f"Input image width ({W}) doesn't match model ({self.img_size[3]})."
        x = self.proj(x)
        if self.flatten:
            B, c, t, d, h, w, = x.shape
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            x = self.norm(x)    # B L C
        else: 
            raise NotImplementedError("You should set 'flatten' argument of PatchEmbed module as True")
        return x




class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        """
        Args:
            dim (int): Number of channels.
            init_values (float): Initial value for the learnable gamma.
            inplace (bool): Whether to perform the operation in-place.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# --- 1D RoPE functions for Temporal Attention ---
def init_1d_positions(length: int, zero_center=False):
    t = torch.arange(length, dtype=torch.float32)
    if zero_center:
        t = t - (length - 1) / 2
    return t

def init_1d_rope_freqs(head_dim: int, num_heads: int, theta: float = 10000.0, rotate: bool = True):
    # Standard 1D RoPE frequency initialization
    freqs_list = []
    # Calculate frequency magnitudes
    # Different theta for time, as it might have different characteristic scales
    mag = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    for _ in range(num_heads):
        if rotate: # For learnable/randomized RoPE, not standard fixed RoPE
            angles = torch.rand(1) * 2 * torch.pi 
            real_part = mag * torch.cos(angles)
            imag_part = mag * torch.sin(angles)
            if head_dim % 2 != 0:
                # This handling of odd head_dim by padding/duplicating might not be standard RoPE
                # Standard RoPE expects head_dim to be even for pairing.
                real_part = torch.cat([real_part, torch.zeros(1)], dim=0)
                imag_part = torch.cat([imag_part, torch.zeros(1)], dim=0)
            else: # This duplication is not standard RoPE; standard uses pairs
                real_part = torch.cat([real_part, real_part], dim=0) # Should be just real_part
                imag_part = torch.cat([imag_part, imag_part], dim=0) # Should be just imag_part

            f_head = torch.cat([real_part, imag_part], dim=0)[:head_dim] # Still unusual for RoPE
            # Standard RoPE: freqs_head = torch.outer(torch.ones(1), mag).float() -> shape (1, head_dim // 2)
            # then freqs = freqs_head.repeat(num_heads, 1, 1)
            # For fixed RoPE, angles should be 0.
            # The original code's init_random_4d_freqs is quite custom.
            # Sticking to a more standard fixed RoPE for simplicity here:
            f_head = mag # shape [head_dim // 2]
            if head_dim % 2 != 0: # Pad if odd
                f_head = torch.cat([f_head, torch.zeros(1)], dim=0)

        else: # Fixed RoPE (standard)
            f_head = mag # shape [head_dim // 2]
            if head_dim % 2 != 0:
                 f_head = torch.cat([f_head, torch.zeros(1)], dim=0)


        freqs_list.append(f_head)
    
    # For standard RoPE, freqs are typically [head_dim // 2] and applied per head implicitly or [num_heads, head_dim // 2]
    # Given the original code structure, let's make it [num_heads, head_dim] where pairs are handled in compute_cis
    # A simpler standard RoPE would have freqs of shape [head_dim // 2]
    
    # Simplified fixed RoPE freqs for 1D:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)) # Shape: [head_dim // 2]
    return freqs # This will be broadcasted across heads or handled appropriately

def compute_1d_rope_cis(freqs_1d, positions_1d, head_dim):
    # freqs_1d: [head_dim // 2]
    # positions_1d: [N_time]
    # Output: [N_time, head_dim] (complex) or [N_time, 1, head_dim] to match q,k head dim
    with torch.cuda.amp.autocast(enabled=False):
        # Outer product: [N_time, head_dim // 2]
        freqs_outer = torch.outer(positions_1d, freqs_1d).float()
        # Create complex numbers: cis(theta) = cos(theta) + i*sin(theta)
        # freqs_cis shape: [N_time, head_dim // 2]
        freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)
        # Reshape to [N_time, 1, head_dim] for broadcasting with q, k [B_eff, num_heads, N_time, head_dim]
        # Each complex number in freqs_cis applies to two real dimensions
        # So, effectively, it should be repeated or handled during qk rotation.
        # Let's prepare it for view_as_complex: [N_time, head_dim//2]
    return freqs_cis # Will be broadcasted to [B_eff, num_heads, N_time, head_dim//2]

# --- 3D RoPE functions for Spatial Attention ---
def init_3d_positions(depth: int, height: int, width: int, zero_center=False):
    total_spatial_tokens = depth * height * width
    s = torch.arange(total_spatial_tokens, dtype=torch.float32)
    
    s_w = (s % width).float()
    s = torch.div(s, width, rounding_mode='floor')
    s_h = (s % height).float()
    s = torch.div(s, height, rounding_mode='floor')
    s_d = s.float()
    
    if zero_center:
        s_w = s_w - (width - 1) / 2
        s_h = s_h - (height - 1) / 2
        s_d = s_d - (depth - 1) / 2
    return s_d, s_h, s_w

# init_3d_positions는 이전 코드의 것을 활용 가능 (D,H,W 좌표 생성)

def init_joint_3d_rope_freqs(head_dim: int, theta: float = 10000.0):
    """
    Joint 3D RoPE를 위한 기본 주파수를 초기화합니다.
    이 주파수는 D, H, W 각 축의 위치에 의해 스케일링되어 합산됩니다.
    """
    # 모든 헤드 차원에 공통적으로 적용될 기본 주파수 세트
    # head_dim의 절반만큼의 주파수 (각 주파수는 복소수의 두 실수부에 해당)
    base_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)) # Shape: [head_dim // 2]
    return base_freqs

def compute_joint_3d_rope_cis(base_freqs, pos_d, pos_h, pos_w, head_dim):
    """
    Joint 3D 위치에 대한 복소수 회전(cisoid) 값을 계산합니다.
    
    Args:
        base_freqs: 기본 주파수 텐서 [head_dim // 2]
        pos_d, pos_h, pos_w: 각 차원별 위치 인덱스 텐서 [N_spatial_tokens]
        head_dim: 어텐션 헤드 차원
        
    Returns:
        freqs_cis: 복소수 회전 텐서 [N_spatial_tokens, head_dim // 2]
    """
    with torch.cuda.amp.autocast(enabled=False):
        # 각 위치 인덱스와 기본 주파수의 외적을 통해 각 축별 각도 성분 계산
        # angles_d/h/w: [N_spatial_tokens, head_dim // 2]
        angles_d = torch.outer(pos_d, base_freqs).float()
        angles_h = torch.outer(pos_h, base_freqs).float() 
        angles_w = torch.outer(pos_w, base_freqs).float()
        
        # 각 축별 각도 성분을 합산하여 최종 각도 계산
        # 다른 theta 값을 사용하고 싶다면, 각 angles_x 에 다른 스케일링 팩터 적용 가능
        # 예: angles_d = torch.outer(pos_d, base_freqs_d) # base_freqs_d 는 다른 theta로 계산
        total_angles = angles_d + angles_h + angles_w # [N_spatial_tokens, head_dim // 2]
        
        # 복소수 회전 값(cisoid) 생성
        freqs_cis = torch.polar(torch.ones_like(total_angles), total_angles)
    return freqs_cis


# Helper function to apply RoPE rotation (common for 1D, 3D components)
def apply_rope_rotation(x, freqs_cis_component):
    # x: [B_eff, num_heads, N, component_head_dim]
    # freqs_cis_component: [N, component_head_dim//2]
    
    # Reshape x for complex multiplication: [B_eff, num_heads, N, component_head_dim//2, 2]
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_) # [B_eff, num_heads, N, component_head_dim//2]
    
    # Prepare freqs_cis for broadcasting: [1, 1, N, component_head_dim//2]
    freqs_cis_b = freqs_cis_component.unsqueeze(0).unsqueeze(0)
    
    # Rotate
    x_rotated_complex = x_complex * freqs_cis_b # Broadcasting happens
    
    # Convert back to real and flatten the last two dims: [B_eff, num_heads, N, component_head_dim]
    x_rotated_real = torch.view_as_real(x_rotated_complex).flatten(3)
    return x_rotated_real.type_as(x)


# --- Factorized Window Partitioning ---
# For temporal attention: input (B_eff, T_seq, C), B_eff = B*D*H*W
def window_partition_temporal(x, temporal_window_size):
    # x: (B_spatial_tubes, T_global, C)
    B_eff, T_global, C = x.shape
    Tw = temporal_window_size
    assert T_global % Tw == 0, f"Global time length must be divisible by temporal window size. T_global:{T_global}, Tw:{Tw}"
    
    x = x.view(B_eff, T_global // Tw, Tw, C)
    windows = x.reshape(-1, Tw, C) # (B_eff * T_global // Tw, Tw, C)
    return windows

def window_reverse_temporal(windows, temporal_window_size, T_global, B_eff):
    # windows: (B_eff * T_global // Tw, Tw, C)
    Tw = temporal_window_size
    C = windows.shape[-1]
    
    x = windows.view(B_eff, T_global // Tw, Tw, C)
    x = x.reshape(B_eff, T_global, C)
    return x

# For spatial attention: input (B_eff, D_seq, H_seq, W_seq, C), B_eff = B*T_global_after_patch_embed
# We can reuse window_partition_4d by setting Tw=1 and input T_dim=1, D_seq, H_seq, W_seq.
# Or create a simpler window_partition_3d
def window_partition_spatial3d(x, spatial_window_size):
    # x: (B_temporal_frames, D_global, H_global, W_global, C)
    B_eff, D, H, W, C = x.shape
    Dw, Hw, Ww = spatial_window_size
    assert D % Dw == 0 and H % Hw == 0 and W % Ww == 0, "Spatial dims must be divisible by window size."

    x = x.view(B_eff, 
               D // Dw, Dw, 
               H // Hw, Hw, 
               W // Ww, Ww, 
               C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous() # B_eff, D//Dw, H//Hw, W//Ww, Dw, Hw, Ww, C
    windows = windows.view(-1, Dw * Hw * Ww, C) # (B_eff * num_spatial_windows, Dw*Hw*Ww, C)
    return windows

def window_reverse_spatial3d(windows, spatial_window_size, D, H, W, B_eff):
    # windows: (B_eff * num_spatial_windows, Dw*Hw*Ww, C)
    Dw, Hw, Ww = spatial_window_size
    C = windows.shape[-1]
    num_spatial_windows_per_frame = (D // Dw) * (H // Hw) * (W // Ww)

    x = windows.view(B_eff, 
                     D // Dw, H // Hw, W // Ww,
                     Dw, Hw, Ww,
                     C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous() # B_eff, D//Dw, Dw, H//Hw, Hw, W//Ww, Ww, C
    x = x.view(B_eff, D, H, W, C)
    return x


class TemporalRoPEWindowAttention(nn.Module):
    def __init__(self, dim, temporal_window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., rope_theta=10000.0, rope_mixed_UNUSED=True, 
                 use_MuTransfer: bool = False,
                 use_flashattn: bool = False,
                 ):
        super().__init__()
        self.dim = dim
        self.temporal_window_size = temporal_window_size # Tw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5 if use_MuTransfer == False else head_dim ** -1 # 250204 jubin added. 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.use_MuTransfer = use_MuTransfer
        self.use_flashattn = use_flashattn

        # Temporal RoPE
        # These positions and frequencies are float, so register_buffer is fine
        rope_positions_t = init_1d_positions(self.temporal_window_size) 
        self.register_buffer('rope_positions_t', rope_positions_t, persistent=False) # persistent=False if not part of state_dict

        rope_freqs_t = init_1d_rope_freqs(head_dim, num_heads, rope_theta, rotate=False) 
        self.register_buffer('rope_freqs_t', rope_freqs_t, persistent=False)
        
        # Precompute CIS for temporal RoPE and store as a non-buffer attribute
        # cis_t shape: [Tw, head_dim//2]
        # Ensure initial computation is on CPU or a default device, then move in forward
        # Or, if these init functions return tensors on default device, it's fine.
        cis_t = compute_1d_rope_cis(rope_freqs_t, rope_positions_t, head_dim)
        self.rope_freqs_cis_t_complex = cis_t # Store as a regular attribute

    def forward(self, x, mask=None):
        B_, N_t, C = x.shape 
        qkv = self.qkv(x).reshape(B_, N_t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flashattn:
            # Move precomputed cis to the current device of q
            # self.rope_freqs_cis_t_complex is [N_t, head_dim//2] (complex)
            device_cis_t = self.rope_freqs_cis_t_complex.to(q.device)

            q = apply_rope_rotation(q, device_cis_t)
            k = apply_rope_rotation(k, device_cis_t)
            if mask is not None:
                nw_t = mask.shape[0]
                mask = torch.cat([mask.to(q.dtype)]*(b_//nw_t)).unsqueeze(1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p, scale=self.scale)  
            x = attn.transpose(1, 2).reshape(B_, N_t, C)
            
        else: 
            q = q * self.scale

            # Move precomputed cis to the current device of q
            # self.rope_freqs_cis_t_complex is [N_t, head_dim//2] (complex)
            device_cis_t = self.rope_freqs_cis_t_complex.to(q.device)
            
            q = apply_rope_rotation(q, device_cis_t)
            k = apply_rope_rotation(k, device_cis_t)
            
            attn = (q @ k.transpose(-2, -1))
            
            if mask is not None: 
                nW_t = mask.shape[0] 
                attn = attn.view(B_ // nW_t, nW_t, self.num_heads, N_t, N_t) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N_t, N_t)
            
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N_t, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# apply_rope_rotation 함수는 이전 코드에서 제공된 것을 그대로 사용합니다.
# def apply_rope_rotation(x, freqs_cis_component): ...

class SpatialRoPEWindowAttention(nn.Module):
    def __init__(self, dim, spatial_window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., rope_theta=10000.0, 
                 use_MuTransfer: bool = False,
                 use_flashattn: bool = False,
                 ):
        super().__init__()
        self.dim = dim
        self.spatial_window_size = spatial_window_size 
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5 if use_MuTransfer == False else head_dim ** -1 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.use_MuTransfer = use_MuTransfer
        self.use_flashattn = use_flashattn

        # Spatial RoPE: positions and base_freqs can be buffers as they are float
        rope_positions_d, rope_positions_h, rope_positions_w = init_3d_positions(
            spatial_window_size[0], spatial_window_size[1], spatial_window_size[2]
        )
        self.register_buffer('rope_positions_d', rope_positions_d, persistent=False)
        self.register_buffer('rope_positions_h', rope_positions_h, persistent=False)
        self.register_buffer('rope_positions_w', rope_positions_w, persistent=False)
        
        spatial_base_freqs = init_joint_3d_rope_freqs(head_dim, rope_theta)
        self.register_buffer('spatial_base_freqs', spatial_base_freqs, persistent=False)
        
        # Precompute Joint 3D CIS tensor and store as a non-buffer attribute
        cis_spatial = compute_joint_3d_rope_cis(
            spatial_base_freqs,
            rope_positions_d, 
            rope_positions_h,
            rope_positions_w,
            head_dim
        )
        self.rope_freqs_cis_spatial_complex = cis_spatial # Store as a regular attribute

    def forward(self, x, mask=None):
        B_, N_s, C = x.shape
        qkv = self.qkv(x).reshape(B_, N_s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        if self.use_flashattn: 
            # Move precomputed cis to the current device of q
            # self.rope_freqs_cis_t_complex is [N_t, head_dim//2] (complex)
            device_cis_spatial = self.rope_freqs_cis_spatial_complex.to(q.device)

            q = apply_rope_rotation(q, device_cis_spatial)
            k = apply_rope_rotation(k, device_cis_spatial)
            if mask is not None:
                nw_s = mask.shape[0]
                mask = torch.cat([mask.to(q.dtype)]*(b_//nw_s)).unsqueeze(1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p, scale=self.scale)  
            x = attn.transpose(1, 2).reshape(B_, N_s, C)
        else: 

            q = q * self.scale

            # Move precomputed cis to the current device of q
            # self.rope_freqs_cis_spatial_complex is [N_s, head_dim // 2] (complex)
            device_cis_spatial = self.rope_freqs_cis_spatial_complex.to(q.device)

            q = apply_rope_rotation(q, device_cis_spatial)
            k = apply_rope_rotation(k, device_cis_spatial)
            
            attn = (q @ k.transpose(-2, -1))

            if mask is not None: 
                nW_s = mask.shape[0] 
                attn = attn.view(B_ // nW_s, nW_s, self.num_heads, N_s, N_s) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N_s, N_s)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N_s, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FactorizedSwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, 
                 temporal_window_size=4, spatial_window_size=(4,7,7), 
                 temporal_shift_size=0, spatial_shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer="GELU", norm_layer=nn.LayerNorm,
                 rope_theta=10000.0, 
                 ls_t_init_values=0.9,
                 ls_s_init_values=0.1,
                 ls_mlp_init_values=1.0,
                 use_MuTransfer=False,
                 use_flashattn=False,
                 use_post_norm=False
                 # Removed rope_mixed, use_rpb, and specific x,y,z,t_ratios for RoPE as they are now factorized
                 # For spatial RoPE, ratios can be internal to SpatialRoPEWindowAttention
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution # (T_in, D_in, H_in, W_in)
        self.num_heads = num_heads
        self.temporal_window_size = temporal_window_size
        self.spatial_window_size = spatial_window_size # (Dw, Hw, Ww)
        self.temporal_shift_size = temporal_shift_size
        self.spatial_shift_size = spatial_shift_size # (Ds, Hs, Ws)
        self.mlp_ratio = mlp_ratio
        self.use_MuTransfer = use_MuTransfer
        self.use_flashattn = use_flashattn
        self.use_post_norm = use_post_norm

        # --- LayerScale Initialization ---
        # init_values가 None이 아닐 경우에만 LayerScale을 적용, 그렇지 않으면 Identity()
        self.ls_t = LayerScale(dim, init_values=ls_t_init_values) if ls_t_init_values is not None else nn.Identity()
        self.ls_s = LayerScale(dim, init_values=ls_s_init_values) if ls_s_init_values is not None else nn.Identity()
        self.ls_mlp = LayerScale(dim, init_values=ls_mlp_init_values) if ls_mlp_init_values is not None else nn.Identity()

        # Temporal Attention Path
        self.norm_t1 = norm_layer(dim)
        self.temporal_attn = TemporalRoPEWindowAttention(
            dim=dim, temporal_window_size=self.temporal_window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            rope_theta=rope_theta,
            use_MuTransfer=use_MuTransfer,
            use_flashattn=use_flashattn 
        )
        self.drop_path_t = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Spatial Attention Path
        self.norm_s1 = norm_layer(dim)
        self.spatial_attn = SpatialRoPEWindowAttention(
            dim=dim, spatial_window_size=self.spatial_window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            rope_theta=rope_theta, # Spatial RoPE might have its own d_ratio,h_ratio settings internally
            use_MuTransfer=use_MuTransfer,
            use_flashattn=use_flashattn
        )
        self.drop_path_s = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP Path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")
        self.drop_path_mlp = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        # Temporal Attention Mask
        if self.temporal_shift_size > 0:
            T_in = self.input_resolution[0]
            temporal_img_mask = torch.zeros((1, T_in, 1)) # B_eff=1, T_in, C_dummy=1
            t_slices = (slice(0, -self.temporal_window_size),
                        slice(-self.temporal_window_size, -self.temporal_shift_size),
                        slice(-self.temporal_shift_size, None))
            cnt = 0
            for t in t_slices:
                temporal_img_mask[:, t, :] = cnt
                cnt += 1
            temporal_mask_windows = window_partition_temporal(temporal_img_mask, self.temporal_window_size) # nW_t, Tw, 1
            temporal_mask_windows = temporal_mask_windows.view(-1, self.temporal_window_size)
            self.temporal_attn_mask = (temporal_mask_windows.unsqueeze(1) - temporal_mask_windows.unsqueeze(2))
            self.temporal_attn_mask = self.temporal_attn_mask.masked_fill(self.temporal_attn_mask != 0, float(-100.0)).masked_fill(self.temporal_attn_mask == 0, float(0.0))
        else:
            self.temporal_attn_mask = None
        
        # Spatial Attention Mask (3D)
        if any(s > 0 for s in self.spatial_shift_size):
            _, D_in, H_in, W_in = self.input_resolution
            spatial_img_mask = torch.zeros((1, D_in, H_in, W_in, 1)) # B_eff=1, D,H,W,C_dummy=1
            
            d_slices = (slice(0, -self.spatial_window_size[0]),
                        slice(-self.spatial_window_size[0], -self.spatial_shift_size[0]),
                        slice(-self.spatial_shift_size[0], None))
            h_slices = (slice(0, -self.spatial_window_size[1]),
                        slice(-self.spatial_window_size[1], -self.spatial_shift_size[1]),
                        slice(-self.spatial_shift_size[1], None))
            w_slices = (slice(0, -self.spatial_window_size[2]),
                        slice(-self.spatial_window_size[2], -self.spatial_shift_size[2]),
                        slice(-self.spatial_shift_size[2], None))
            cnt = 0
            for d_sl in d_slices:
                for h_sl in h_slices:
                    for w_sl in w_slices:
                        spatial_img_mask[:, d_sl, h_sl, w_sl, :] = cnt
                        cnt += 1
            
            spatial_mask_windows = window_partition_spatial3d(spatial_img_mask, self.spatial_window_size) # nW_s, Dw*Hw*Ww, 1
            spatial_mask_windows = spatial_mask_windows.view(-1, self.spatial_window_size[0] * self.spatial_window_size[1] * self.spatial_window_size[2])
            self.spatial_attn_mask = (spatial_mask_windows.unsqueeze(1) - spatial_mask_windows.unsqueeze(2))
            self.spatial_attn_mask = self.spatial_attn_mask.masked_fill(self.spatial_attn_mask != 0, float(-100.0)).masked_fill(self.spatial_attn_mask == 0, float(0.0))
        else:
            self.spatial_attn_mask = None
        
        # Register buffers if they are not None
        if self.temporal_attn_mask is not None:
            self.register_buffer("temporal_attn_mask_buffer", self.temporal_attn_mask)
        else: # Make it a non-None buffer so it's always present
            self.register_buffer("temporal_attn_mask_buffer", torch.zeros(0)) # Or handle None in forward

        if self.spatial_attn_mask is not None:
            self.register_buffer("spatial_attn_mask_buffer", self.spatial_attn_mask)
        else:
            self.register_buffer("spatial_attn_mask_buffer", torch.zeros(0))


    def forward(self, x):
        B, L, C = x.shape
        T_in, D_in, H_in, W_in = self.input_resolution
        assert L == T_in * D_in * H_in * W_in, f"Input feature has wrong size, L:{L}, T_in:{T_in}, D_in:{D_in}, H_in:{H_in}, W_in:{W_in}"

        # --- Temporal Attention ---
        shortcut_t = x
        x_norm_t = self.norm_t1(x)
        x_reshaped_for_temporal = x_norm_t.view(B, T_in, D_in, H_in, W_in, C)
        
        # Permute for temporal attention: (B, D, H, W, T, C) -> (B*D*H*W, T, C)
        x_spatial_tubes = x_reshaped_for_temporal.permute(0, 2, 3, 4, 1, 5).contiguous()
        B_eff_t = B * D_in * H_in * W_in
        x_spatial_tubes = x_spatial_tubes.view(B_eff_t, T_in, C)

        # Temporal Shift
        if self.temporal_shift_size > 0:
            shifted_x_temporal = torch.roll(x_spatial_tubes, shifts=(-self.temporal_shift_size,), dims=1)
        else:
            shifted_x_temporal = x_spatial_tubes
        
        # Temporal Window Partition
        # (B_eff_t * T_in // Tw, Tw, C)
        temporal_windows = window_partition_temporal(shifted_x_temporal, self.temporal_window_size) 
        
        # Temporal Attention
        # Use self.temporal_attn_mask_buffer if not None, else None
        current_temporal_mask = self.temporal_attn_mask_buffer if self.temporal_attn_mask_buffer.numel() > 0 else None
        if current_temporal_mask is not None: current_temporal_mask = current_temporal_mask.to(temporal_windows.device)

        attn_temporal_windows = self.temporal_attn(temporal_windows, mask=current_temporal_mask)

        # Temporal Window Reverse
        # (B_eff_t, T_in, C)
        shifted_x_temporal = window_reverse_temporal(attn_temporal_windows, self.temporal_window_size, T_in, B_eff_t)

        # Temporal Reverse Shift
        if self.temporal_shift_size > 0:
            x_spatial_tubes = torch.roll(shifted_x_temporal, shifts=(self.temporal_shift_size,), dims=1)
        else:
            x_spatial_tubes = shifted_x_temporal
            
        # Reshape back to (B, T, D, H, W, C)
        x_after_temporal = x_spatial_tubes.view(B, D_in, H_in, W_in, T_in, C).permute(0, 4, 1, 2, 3, 5).contiguous()
        x_after_temporal = x_after_temporal.view(B, L, C) # Flatten back for residual
        # Apply LayerScale before DropPath and residual connection
        x = shortcut_t + self.drop_path_t(self.ls_t(x_after_temporal))

        # --- Spatial Attention ---
        shortcut_s = x
        x_norm_s = self.norm_s1(x)
        x_reshaped_for_spatial = x_norm_s.view(B, T_in, D_in, H_in, W_in, C)

        # Permute for spatial attention: (B, T, D, H, W, C) -> (B*T, D, H, W, C)
        x_temporal_frames = x_reshaped_for_spatial.permute(0, 1, 2, 3, 4, 5).contiguous() # Already B,T,D,H,W,C
        B_eff_s = B * T_in
        x_temporal_frames = x_temporal_frames.view(B_eff_s, D_in, H_in, W_in, C)

        # Spatial Shift
        if any(s > 0 for s in self.spatial_shift_size):
            shifted_x_spatial = torch.roll(x_temporal_frames, 
                                           shifts=(-self.spatial_shift_size[0], -self.spatial_shift_size[1], -self.spatial_shift_size[2]),
                                           dims=(1, 2, 3))
        else:
            shifted_x_spatial = x_temporal_frames
            
        # Spatial Window Partition
        # (B_eff_s * num_spatial_windows, Dw*Hw*Ww, C)
        spatial_windows = window_partition_spatial3d(shifted_x_spatial, self.spatial_window_size)
        
        # Spatial Attention
        current_spatial_mask = self.spatial_attn_mask_buffer if self.spatial_attn_mask_buffer.numel() > 0 else None
        if current_spatial_mask is not None: current_spatial_mask = current_spatial_mask.to(spatial_windows.device)
        
        attn_spatial_windows = self.spatial_attn(spatial_windows, mask=current_spatial_mask)

        # Spatial Window Reverse
        # (B_eff_s, D_in, H_in, W_in, C)
        shifted_x_spatial = window_reverse_spatial3d(attn_spatial_windows, self.spatial_window_size, D_in, H_in, W_in, B_eff_s)

        # Spatial Reverse Shift
        if any(s > 0 for s in self.spatial_shift_size):
            x_temporal_frames = torch.roll(shifted_x_spatial,
                                           shifts=(self.spatial_shift_size[0], self.spatial_shift_size[1], self.spatial_shift_size[2]),
                                           dims=(1, 2, 3))
        else:
            x_temporal_frames = shifted_x_spatial
            
        # Reshape back to (B, T*D*H*W, C)
        x_after_spatial = x_temporal_frames.view(B, T_in * D_in * H_in * W_in, C)
        # Apply LayerScale before DropPath and residual connection
        x = shortcut_s + self.drop_path_s(self.ls_s(x_after_spatial))
        
        # --- MLP ---
        x = x + self.drop_path_mlp(self.mlp(self.norm2(x)))
        x = x + self.drop_path_mlp(self.ls_mlp(self.mlp(self.norm2(x))))
        
        return x

# Update RoPE4DBasicLayer to use FactorizedSwinTransformerBlock
# and to pass factorized window/shift sizes
class RoPE4DBasicLayer(nn.Module):
    def __init__(self, 
                 dim, 
                 input_resolution, # (T,D,H,W)
                 depth, 
                 num_heads, 
                 temporal_window_size=4, 
                 spatial_window_size=(4,4,4), # (Dw,Hw,Ww)
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 norm_layer=nn.LayerNorm, 
                 downsample: Optional[nn.Module] = None,
                 upsample: Optional[nn.Module] = None,
                 use_checkpoint=False,
                 rope_theta=10000.0,
                 ls_t_init_values=0.9,
                 ls_s_init_values=0.1,
                 ls_mlp_init_values=1.0,
                 use_MuTransfer=False,
                 use_flashattn=False,
                 use_post_norm=False,
                 # Removed RoPE specific ratios as they are handled internally by factorized attentions
                 **kwargs # To catch unused old RoPE params from main model if any
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Determine shift sizes for temporal and spatial
            # Temporal shift for odd blocks
            current_temporal_shift_size = temporal_window_size // 2 if i % 2 == 1 else 0
            # Spatial shift for odd blocks
            current_spatial_shift_size = tuple(sw // 2 if i % 2 == 1 else 0 for sw in spatial_window_size)

            # Adjust window sizes if they exceed input resolution
            current_temporal_window_size = min(input_resolution[0], temporal_window_size)
            current_spatial_window_size = tuple(min(r, w) for r, w in zip(input_resolution[1:], spatial_window_size))

            if input_resolution[0] <= current_temporal_window_size: # if temporal dim is too small
                current_temporal_shift_size = 0 # No shift if window covers full dim
                current_temporal_window_size = input_resolution[0]

            if min(input_resolution[1:]) <= min(current_spatial_window_size): # if any spatial dim is too small
                current_spatial_shift_size = (0,0,0) # No shift
                # Ensure spatial window size does not exceed current spatial resolution
                current_spatial_window_size = tuple(min(r_dim, w_dim) for r_dim, w_dim in zip(input_resolution[1:], current_spatial_window_size))


            self.blocks.append(
                FactorizedSwinTransformerBlock(
                    dim=dim, 
                    input_resolution=input_resolution,
                    num_heads=num_heads, 
                    temporal_window_size=current_temporal_window_size,
                    spatial_window_size=current_spatial_window_size,
                    temporal_shift_size=current_temporal_shift_size,
                    spatial_shift_size=current_spatial_shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    drop=drop, 
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    rope_theta=rope_theta,
                    ls_t_init_values=ls_t_init_values, # LayerScale 초기값 전달
                    ls_s_init_values=ls_s_init_values,
                    ls_mlp_init_values=ls_mlp_init_values,
                )
            )

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(input_resolution=input_resolution, dim=dim, norm_layer=norm_layer) # PatchMergingV2
        self.upsample = upsample
        if self.upsample is not None: 
            self.upsample = nn.Sequential(
                nn.Linear(dim,dim*(2**3),bias=True),
                upsample(spatial_scale=2, temporal_scale=1),
                nn.Linear(dim,int(dim*0.5),bias=True),
            )

    
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RoPE4DSwinTransformer_Downstream(nn.Module):
    """
    RoPE-enhanced 4D Swin Transformer for Downstream Tasks

    Based on Ver11 (Simmim_RoPE4DSwinTransformer) but modified for downstream tasks:
    - Removed masking mechanism
    - Removed decoder
    - Direct forward pass without mask requirement

    Key features from Ver11:
    - Factorized attention (Temporal + Spatial)
    - Rotary Position Embedding (RoPE)
    - LayerScale for training stability
    """

    def __init__(
        self,
        img_size: Tuple,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,   
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 4,
        c_multiplier: int = 2,
        downsample="mergingv2",
        upsample='pixelshuffle',
        num_classes=2,
        rope_theta=10000.0,           
        ls_t_init_values=0.9,
        ls_s_init_values=0.1,
        ls_mlp_init_values=1.0,
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
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.img_size = change_st_to_ts_dimension(self.img_size) # (D, H, W, T) -> (T, D, H, W)
        self.patch_size = change_st_to_ts_dimension(patch_size)  # (Pt, Pd, Ph, Pw) -> (Pd, Ph, Pw, Pt)
        self.in_chans = in_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.to_float = to_float

        self.window_size = window_size
        temporal_window_size = window_size[-1]  # window_size = (D, H, W, T)
        spatial_window_size = window_size[:-1]  # window_size = (D, H, W, T)

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            flatten=True, # 250529, changed to True again.
        )
        patches_resolution = self.patch_embed.patches_resolution # (T_patched, D_patched, H_patched, W_patched)
        self.patches_resolution = patches_resolution

        self.use_MuTransfer = kwargs.get('use_MuTransfer', False)
        self.use_flashattn = kwargs.get('use_flashattn', False)
        self.use_post_norm = kwargs.get('use_post_norm', False)

        if self.use_flashattn:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                print("Flash Attention is not supported on this device. Using normal attention instead.")

        #patch_num = int((img_size[0]/patch_size[0]) * (img_size[1]/patch_size[1]) * (img_size[2]/patch_size[2]))
        #time_num = int(img_size[3]/patch_size[3])
        #print img, patch size, patch dim
        print("img_size: ", img_size)
        print("patch_size: ", patch_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Calculate input resolution for the current layer
            # PatchMergingV2 in this code merges D,H,W by 2.
            current_input_resolution = (
                patches_resolution[0],
                patches_resolution[1] // (2 ** i_layer),
                patches_resolution[2] // (2 ** i_layer),
                patches_resolution[3] // (2 ** i_layer),

            )
            current_dim = int(embed_dim * 2 ** i_layer)

            # Adjust window sizes if they exceed current_input_resolution for this layer
            # This logic is now moved into RoPE4DBasicLayer's block creation loop
            
            layer = RoPE4DBasicLayer(
                dim=current_dim,
                input_resolution=current_input_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                temporal_window_size=temporal_window_size, # Pass base T-window size
                spatial_window_size=spatial_window_size, # Pass base S-window size (Dw,Hw,Ww)
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=down_sample_mod if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                rope_theta=rope_theta,
                ls_t_init_values=ls_t_init_values,
                ls_s_init_values=ls_s_init_values,
                ls_mlp_init_values=ls_mlp_init_values, 
                use_MuTransfer=self.use_MuTransfer,
                use_flashattn=self.use_flashattn,
                use_post_norm=self.use_post_norm
            )
            self.layers.append(layer)
        
        # The last layer's output resolution, after all merging
        self.last_layer_resolution = (
                patches_resolution[0],
                patches_resolution[1] // 2**(self.num_layers-1),
                patches_resolution[2] // 2**(self.num_layers-1),
                patches_resolution[3] // 2**(self.num_layers-1),

            )


        encoder_out_dim = int(embed_dim * c_multiplier ** (self.num_layers - 1))

        self.norm_layer_class = norm_layer
        self.norm = norm_layer(encoder_out_dim)
        # Removed mask_token - not needed for downstream tasks
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        self.encoder_out_patch_num = (
            patches_resolution[0],
            patches_resolution[1] // (2 ** (self.num_layers-1)),
            patches_resolution[2] // (2 ** (self.num_layers-1)),
            patches_resolution[3] // (2 ** (self.num_layers-1)),
        )
        # decoder specific 
        self.decoder_type = 'mlp'  
        if self.decoder_type == 'linear':
            spatial_expansion_multiplier = int(self.patch_size[1] * 2**3)   # 3 stages of patch merging
            self.decoder_embed = nn.Linear(encoder_out_dim, (spatial_expansion_multiplier**3) * self.patch_size[0], bias=True)
            self.decoder_expand = PixelShuffle4D(spatial_scale=spatial_expansion_multiplier, temporal_scale=self.patch_size[0])

        elif self.decoder_type == 'mlp': 
            self.drop_rate = drop_rate
            # temporal expansion
            self.temporal_expansion_multiplier = int(self.img_size[0] // self.encoder_out_patch_num[0])
            self.decoder_temporal_in_dim = encoder_out_dim
            self.decoder_temporal_out_dim = encoder_out_dim * self.temporal_expansion_multiplier
            self.decoder_temporal_emb = nn.Sequential(
                nn.Linear(self.decoder_temporal_in_dim,  int(self.decoder_temporal_in_dim * self.mlp_ratio), bias=True),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(int(self.decoder_temporal_in_dim * self.mlp_ratio),  self.decoder_temporal_in_dim, bias=True),
                nn.Dropout(drop_rate), 
                norm_layer(self.decoder_temporal_in_dim)
            )
            self.decoder_temporal_expand = nn.Linear(self.decoder_temporal_in_dim, self.decoder_temporal_out_dim, bias=True)
            self.decoder_temporal_upsample = PixelShuffle4D(spatial_scale=1, temporal_scale=self.temporal_expansion_multiplier)
            
            # spatial expansion
            self.spatial_expansion_multiplier = int(self.patch_size[1] * 2**3)   # 3 stages of patch merging
            self.decoder_spatial_in_dim = encoder_out_dim
            self.decoder_spatial_out_dim = self.spatial_expansion_multiplier**3
            self.decoder_spatial_emb = nn.Sequential(
                nn.Linear(self.decoder_spatial_in_dim, int(self.decoder_spatial_in_dim * self.mlp_ratio), bias=True),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(int(self.decoder_spatial_in_dim * self.mlp_ratio), self.decoder_spatial_in_dim, bias=True),
                nn.Dropout(drop_rate), 
                norm_layer(self.decoder_spatial_in_dim)
            )
            self.decoder_spatial_expand = nn.Linear(self.decoder_spatial_in_dim, self.decoder_spatial_out_dim, bias=True)
            self.decoder_spatial_upsample = PixelShuffle4D(spatial_scale=self.spatial_expansion_multiplier, temporal_scale=1)

        elif self.decoder_type == 'reverse_swin': 
            # inverse swin-T
            decoder_depths = [2, 6, 2, 2]
            decoder_num_heads = [24, 12, 6, 3]
            decoder_c_multiplier = 0.5
            decoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]
            up_sample_mod = look_up_option(upsample, EXPANDING_MODE) if isinstance(upsample, str) else upsample


            self.decoder_layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                current_input_resolution = (
                    self.encoder_out_patch_num[0],
                    self.encoder_out_patch_num[1] * (2 ** i_layer),
                    self.encoder_out_patch_num[2] * (2 ** i_layer),
                    self.encoder_out_patch_num[3] * (2 ** i_layer),
                )
                current_dim =  int(encoder_out_dim * (0.5 ** i_layer))

                layer = RoPE4DBasicLayer(
                    dim=current_dim,
                    input_resolution=current_input_resolution,
                    depth=decoder_depths[i_layer],
                    num_heads=decoder_num_heads[i_layer],
                    temporal_window_size=temporal_window_size, # Pass base T-window size
                    spatial_window_size=spatial_window_size, # Pass base S-window size (Dw,Hw,Ww)
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale,
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=decoder_dpr[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample= None,
                    upsample='',
                    use_checkpoint=use_checkpoint,
                    rope_theta=rope_theta,
                    ls_t_init_values=ls_t_init_values,
                    ls_s_init_values=ls_s_init_values,
                    ls_mlp_init_values=ls_mlp_init_values, 
                    use_MuTransfer=self.use_MuTransfer,
                    use_flashattn=self.use_flashattn,
                    use_post_norm=self.use_post_norm
                )
                self.decoder_layers.append(layer)


            decoder_out_dim = int(encoder_out_dim * (0.5 ** (self.num_layers - 1)))
            self.decoder_temporal_emb = nn.Linear(decoder_out_dim, decoder_out_dim * self.patch_size[-1], bias=True)
            self.decoder_temporal_expand = PixelShuffle4D(spatial_scale=1, temporal_scale=self.patch_size[-1])
            self.decoder_spatial_emb = nn.Linear(decoder_out_dim, (self.patch_size[0]**3), bias=True)
            self.decoder_spatial_expand = PixelShuffle4D(spatial_scale=self.patch_size[0], temporal_scale=1)

        # --------------------------------------------------------------------------
        # weight initialize 
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


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
        if self.decoder_type == 'w/ape':
            print("MuP Used")
            in_dim = self.decoder_spatial_embed.in_features
            out_dim = self.decoder_spatial_embed.out_features
            from mup import MuReadout
            self.decoder_spatial_embed = MuReadout(in_dim, out_dim, bias=True) # 250208 jubin changed. use bias=True
        
        elif self.decoder_type == 'mlp': 
            print("MuP Used")
            drop_rate = self.drop_rate
            temporal_in_dim = self.decoder_temporal_in_dim
            temporal_out_dim = self.decoder_temporal_out_dim
            spatial_in_dim = self.decoder_spatial_in_dim
            spatial_out_dim = self.decoder_spatial_out_dim
            from mup import MuReadout
            self.decoder_temporal_emb = nn.Sequential(
                nn.Linear(temporal_in_dim,  int(temporal_in_dim * self.mlp_ratio), bias=True),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(int(temporal_in_dim * self.mlp_ratio),  temporal_in_dim, bias=True),
                nn.Dropout(drop_rate), 
                self.norm_layer_class(temporal_in_dim)
            )
            self.decoder_temporal_expand = MuReadout(temporal_in_dim, temporal_out_dim, bias=True)
            
            # spatial expansion
            self.decoder_spatial_emb = nn.Sequential(
                MuReadout(spatial_in_dim, int(spatial_out_dim * self.mlp_ratio), bias=True),
                nn.GELU(),
                nn.Dropout(drop_rate),
                MuReadout(int(spatial_out_dim * self.mlp_ratio), spatial_in_dim, bias=True),
                nn.Dropout(drop_rate), 
                self.norm_layer_class(spatial_in_dim)
            )
            self.decoder_spatial_expand = MuReadout(spatial_in_dim, spatial_out_dim, bias=True)
        else: 
            raise NotImplementedError("MuParameterization now only support setting decoder type as 'w/ape'")

    def patchify(self, imgs):
        spatial_p = self.patch_size[0]
        temporal_p = self.patch_size[3]

        assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] \
            and imgs.shape[2] % spatial_p == 0 \
            and imgs.shape[5] % temporal_p == 0 
        
        d = h = w = imgs.shape[2] // spatial_p   
        imgs = rearrange(imgs, 'B C (d p1) (h p2) (w p3) (t p4) -> B (C p1 p2 p3 p4) d h w t', 
                         B=imgs.shape[0], C=self.in_chans,  d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2], t=self.grid_size[3],
                         p1=spatial_p, p2=spatial_p, p3=spatial_p, p4=temporal_p)


        return imgs 


    def unpatchify(self, z):
        spatial_p = self.patch_size[0]
        temporal_p = self.patch_size[3]

        B, _, d, h, w, t = z.shape 
        C = self.in_chans
        imgs = rearrange(z, 'B (C p1 p2 p3 p4) d h w t -> B C (d p1) (h p2) (w p3) (t p4)',
                         B=B, C=C, d=d, h=h, w=w, t=t,
                         p1=spatial_p, p2=spatial_p, p3=spatial_p, p4=temporal_p)


        return imgs     



    def forward_encoder(self, x: torch.Tensor):
        """
        Encoder forward pass for downstream tasks (no masking)

        Args:
            x: Input tensor (B, C, T, D, H, W)
        Returns:
            x: Encoded features (B, C, D*H*W*T) matching Ver9 format
        """
        # patch embedding
        x = self.patch_embed(x)     # B, L_patched, C_embed_dim
                                    # L_patched = T_p * D_p * H_p * W_p

        # No masking for downstream tasks - use all patches

        # swin blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B, L_final, C_final

        # Reshape to (B, C, D, H, W, T) then flatten spatial-temporal dims
        # to match Ver9 output format for decoder compatibility
        b, _, c = x.shape
        t, d, h, w = self.last_layer_resolution
        x = x.permute(0, 2, 1).reshape(b, c, d, h, w, t)

        # for decoder: flatten spatial-temporal dims
        x = x.flatten(start_dim=2)  # B, C, D*H*W*T

        return x  # B, C, D*H*W*T (matches Ver9 format)


    def forward_decoder(self, x: torch.Tensor): 
        if self.decoder_type == 'mlp': 
            B, C, d, h, w, t = x.shape
            # temporal expansion
            x = x.flatten(2).permute(0, 2, 1)    # B L C
            x = self.decoder_temporal_emb(x)    # B L C
            after_temporal_emb = x.shape 
            x = self.decoder_temporal_expand(x) # B L C 
            after_temporal_expansion = x.shape 
            x = x.reshape(B, d, h, w, t, -1)
            x = self.decoder_temporal_upsample(x) # B d h w t*pt C
            after_up_sample = x.shape 
            T = t * self.temporal_expansion_multiplier
            x = x.permute(0, 5, 1, 2, 3, 4) # B C d h w T 
            # spatial expansion
            x = x.flatten(2).permute(0, 2, 1)   # B L C
            x = self.decoder_spatial_emb(x)
            x = self.decoder_spatial_expand(x)
            x = x.reshape(B, d, h, w, T, -1)
            x = self.decoder_spatial_upsample(x)    # B d*pd h*ph w*pw T C 
            x =  x.permute(0, 5, 1, 2, 3, 4) # B C D H W T 
           
            
        """
        if self.decoder_type == 'linear':
            B = x.shape[0]
            x = self.decoder_embed(x)   # B L C
            D, H, W, T = self.encoder_out_patch_num
            x = self.decoder_expand(x.reshape(B, D, H, W, T, -1))  # B D H W T C
            x = x.permute(0, 5, 1, 2, 3, 4) # B C D H W T 
            # 250219 jubin. in frontier code it returns x directly without permutation
            
        elif self.decoder_type == 'w/ape': 
            B, C, D, H, W, t = x.shape  
            x = x.flatten(2).permute(0, 2, 1)    # B L C
            x = self.decoder_temporal_emb(x) # B L C*pt
            x = self.decoder_temporal_mlp(x)
            x = x.permute(0, 2, 1).reshape(B, -1, D, W, H, t)
            x = self.decoder_temporal_expand(x) # B C D H W T*pt
            T = t * self.temporal_expansion_multiplier
            x = self.decoder_pos_embed(x)   # B C D H W T*pt
            x = x.flatten(2).permute(0, 2, 1)   # B L C
            x = self.decoder_spatial_embed(x)   # B L -1
            x = self.decoder_spatial_expand(x.permute(0, 2, 1).reshape(B, -1, D, W, H, T))  # B C D H W T 

        elif self.decoder_type == 'reverse_swin':
            # swin blocks 
            for i in range(self.num_layers):
                x = self.decoder_pos_embeds[i](x)
                x = self.decoder_layers[i](x.contiguous())  # B C D H W T 
            # temporal expansion
            B, C, D, H, W, t = x.shape  
            x = x.flatten(2).permute(0, 2, 1)    # B L C
            x = self.decoder_temporal_emb(x) # B L C*pt
            x = self.decoder_temporal_expand(x.reshape(B, D, H, W, t, -1)) # B D H W T*pt 
            x = x.permute(0, 5, 1, 2, 3, 4) # B C D H W T*pt
            T = x.shape[-1]
            x = x.flatten(2).permute(0, 2, 1)   # B L C
            x = self.decoder_spatial_emb(x)   # B L -1
            x = self.decoder_spatial_expand(x.reshape(B, D, H, W, T, -1))  # B D H W T C
            x = x.permute(0, 5, 1, 2, 3, 4) # B C D H W T  
        """
        return x 


    def forward(self, x: torch.Tensor):
        """
        Forward pass for downstream tasks

        Args:
            x: Input tensor (B, C, D, H, W, T)
        Returns:
            z: Encoded features (B, C, D, H, W, T)
        """
        # Input x is (B, C, D, H, W, T_in_original_img_size)
        # Need to convert to (B, C, T, D, H, W) for encoder
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B, C, T_img, D_img, H_img, W_img

        if self.to_float:
            x = x.float()

        z = self.forward_encoder(x)   # B, C, D, H, W, T

        # Return encoded features for downstream task head
        return z  # B, C, D, H, W, T 

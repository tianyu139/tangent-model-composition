import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
            resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn
from timm.layers.helpers import to_2tuple
from models.linear_layers import LinearLinear, LinearSequential

EPS = 1e-8

class LinearAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        assert qk_norm is False

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # assert self.fused_attn is False

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = LinearLinear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = LinearLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #self.linear_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)


    def forward(self, input, input_jvp=None):
        B, N, C = input.shape
        # Original attention output
        x = input
        x_jvp = input_jvp

        qkv, qkv_jvp = self.qkv(x, x_jvp)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv_jvp = qkv_jvp.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q_jvp, k_jvp, v_jvp = qkv_jvp.unbind(0)

        if self.qk_norm:
            q, q_jvp = self.q_norm(q, q_jvp)
            k, k_jvp = self.k_norm(k, k_jvp)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)

        q_jvp = q_jvp * self.scale
        A_jvp = (q_jvp) @ k.transpose(-2, -1)
        B_jvp = q @ (k_jvp).transpose(-2, -1)
        C_jvp = attn @ (v_jvp)

        # x is output of attention
        #x_copy = input
        # jvp_prev
        #if input_jvp is not None:
            # New shape: 3 x B x Heads x N x D
        #    qkv_t2 = F.linear(input_jvp, self.qkv.weight, None).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        #    q_t2, k_t2, v_t2 = qkv_t2.unbind(0)

        #qkv_t3 = self.linear_qkv(x_copy).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        #q_t3, k_t3, v_t3 = qkv_t3.unbind(0)

#        if input_jvp is not None:
#            q_t2 = q_t2 * self.scale
#            q_t3 = q_t3 * self.scale
#
#            A_jvp = (q_t3 + q_t2) @ k.transpose(-2, -1)
#            B_jvp = q @ (k_t3 + k_t2).transpose(-2, -1)
#            C_jvp = attn @ (v_t2 + v_t3)
#        else:
#            q_t3 = q_t3 * self.scale
#
#            A_jvp = (q_t3) @ k.transpose(-2, -1)
#            B_jvp = q @ (k_t3).transpose(-2, -1)
#            C_jvp = attn @ (v_t3)

        dm_by_dr = A_jvp + B_jvp
        assert dm_by_dr.shape == attn.shape

        attn_filter_1 = dm_by_dr * attn
        attn_filter_2 = (torch.diagonal(dm_by_dr @ attn.transpose(-2, -1), dim1=-2, dim2=-1).unsqueeze(dim=-1)) * attn
        attn_filter = attn_filter_1 - attn_filter_2

        # Approximate attention derivative w identity
        # x_jvp = (A_jvp + B_jvp) @ v + C_jvp

        # Approximate attention derivative w identity
        # x_jvp = (A_jvp + B_jvp) @ v + C_jvp
        x_jvp = attn_filter @ v + C_jvp
        x_jvp = x_jvp.transpose(1, 2).reshape(B, N, C)

        x, x_jvp = self.proj(x, x_jvp)
        # x_jvp = F.linear(x_jvp, self.proj.weight, None) + self.linear_proj(x)

        x = self.proj_drop(x)
        x_jvp = self.proj_drop(x_jvp)

        return x, x_jvp


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LinearGELU(nn.GELU):

    def __init__(self):
        super().__init__()

    def forward(self, input, input_jvp):
        output = super().forward(input)
        gauss = torch.distributions.normal.Normal(0, 1)

        output_jvp = input_jvp * ( output / (input + EPS) + input * torch.exp(gauss.log_prob(input)))

        return output, output_jvp

class LinearMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=LinearGELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()

        assert use_conv is False
        assert norm_layer is None
        assert act_layer == LinearGELU

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else LinearLinear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = LinearGELU()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])


    def forward(self, input, input_jvp=None):
        x, x_jvp = input, input_jvp

        x, x_jvp = self.fc1(x, x_jvp)
        x, x_jvp = self.act(x, x_jvp)
        x = self.drop1(x)
        x_jvp = self.drop1(x_jvp)
        x, x_jvp = self.fc2(x, x_jvp)
        x = self.drop2(x)
        x_jvp = self.drop2(x_jvp)

        return x, x_jvp


class LinearLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

        assert len(self.normalized_shape) == 1

        if self.elementwise_affine:
            self.linear_weight = nn.Parameter(torch.zeros_like(self.weight, **factory_kwargs, requires_grad=True))
            self.linear_bias = nn.Parameter(torch.zeros_like(self.bias, **factory_kwargs, requires_grad=True))
        else:
            self.register_parameter('linear_weight', None)
            self.register_parameter('linear_bias', None)


    def forward(self, input, input_jvp):
        x = input
        output = super().forward(x)

        dim = -1
        x_jvp = input_jvp
        output_jvp_t1 = F.layer_norm(input, self.normalized_shape, self.linear_weight, self.linear_bias, self.eps)

        mean_x = torch.mean(x, dim=dim, keepdim=True)
        std_x = torch.sqrt(torch.square(x - mean_x).mean(dim=dim, keepdim=True) + self.eps)
        mean_x_jvp = torch.mean(x_jvp, dim=dim, keepdim=True)

        output_jvp_a = (x_jvp - torch.mean(x_jvp, dim=dim, keepdim=True)) * std_x
        corr = 2 * torch.mean((x - mean_x) * (x_jvp - mean_x_jvp), dim=dim, keepdim=True)
        output_jvp_b = 0.5 / std_x * corr * (input - mean_x)
        output_jvp_c = torch.square(1 / std_x) * (output_jvp_a - output_jvp_b)
        output_jvp_t2 = output_jvp_c * self.weight

        output_jvp = output_jvp_t1 + output_jvp_t2

        return output, output_jvp


    def _my_layer_norm(x, dim, eps=1e-5):
        # For reference purpoises
        mean = torch.mean(x, dim=dim, keepdim=True)
        var = torch.square(x - mean).mean(dim=dim, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)


class LinearBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=LinearGELU,
            norm_layer=LinearLayerNorm
    ):
        super().__init__()

        assert norm_layer == LinearLayerNorm or norm_layer.func == LinearLayerNorm

        self.norm1 = norm_layer(dim)
        self.attn = LinearAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        assert init_values is None
        assert drop_path == 0.0

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = LinearMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input, input_jvp):
        #x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        #x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = input
        x_jvp = input_jvp

        x, x_jvp = self.norm1(x, x_jvp)
        x, x_jvp = self.attn(x, x_jvp)

        x = x + input
        x_jvp = x_jvp + input_jvp

        input2 = x
        input2_jvp = x_jvp

        x, x_jvp = self.norm2(x, x_jvp)
        x, x_jvp = self.mlp(x, x_jvp)

        x = x + input2
        x_jvp = x_jvp + input2_jvp

        output = x
        output_jvp = x_jvp

        return output, output_jvp


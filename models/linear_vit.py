import logging
import math
import copy
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
from models.linear_attention_layers import *


EPS = 1e-8


class Linear_Vision_Transformer(nn.Module):
    def __init__(
        self,
        replace_cls_token,
        replace_cls_token_layer,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: str = '',
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = LinearBlock,
    ):
        super().__init__()

        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        assert block_fn == LinearBlock

        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(LinearLayerNorm, eps=1e-6)
        act_layer = act_layer or LinearGELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.replace_cls_token = replace_cls_token
        self.replace_cls_token_layer = replace_cls_token_layer

        if class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            if self.replace_cls_token:
                self.new_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                torch.nn.init.normal_(self.new_cls_token, std=1e-6)
                self.linear_new_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = LinearSequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) if not use_fc_norm else None
        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else None
        self.head_drop = nn.Dropout(drop_rate)
        assert num_classes > 0
        self.head = LinearLinear(self.embed_dim, num_classes)

    def forward(self, x, x_jvp=None):
        assert x_jvp is None
        x = self.patch_embed(x)
        x_jvp = torch.zeros_like(x)

        x, x_jvp = self._pos_embed(x, x_jvp)
        x = self.patch_drop(x)
        x_jvp = self.patch_drop(x_jvp)

        if isinstance(self.norm_pre, nn.Identity):
            x = self.norm_pre(x)
            x_jvp = torch.zeros_like(x)
        else:
            x_jvp = torch.zeros_like(x)
            x, x_jvp = self.norm_pre(x, x_jvp)

        for block_idx, block in enumerate(self.blocks):
            if self.replace_cls_token and block_idx == self.replace_cls_token_layer:
                B = len(x)
                cls_tokens = self.new_cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x[:,1:]), dim=1)

                linear_cls_tokens = self.linear_new_cls_token.expand(B, -1, -1)
                x_jvp = torch.cat((linear_cls_tokens, x_jvp[:,1:]), dim=1)

            x, x_jvp = block(x, x_jvp)

        if self.norm is not None:
            x, x_jvp = self.norm(x, x_jvp)

        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
            x_jvp = x_jvp[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x_jvp[:, 0]

        if self.fc_norm is not None:
            x, x_jvp = self.fc_norm(x, x_jvp)

        x = self.head_drop(x)
        x_jvp = self.head_drop(x_jvp)
        x, x_jvp = self.head(x, x_jvp)

        return x, x_jvp


    def load_pretrained_weights(self, state_dict, load_fc=False, deletes=[]):
        fc_keys = ['head.weight', 'head.bias']
        if not load_fc:
            state_dict = copy.deepcopy(state_dict)
            deletes = copy.deepcopy(deletes)
            for k in fc_keys:
                if k in state_dict:
                    del state_dict[k]
                deletes.append(k)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0
        assert all(['linear_' in m or 'new_cls_token' in m or m in deletes for m in missing])

        print("VIT: pretrained weights loaded")

    def _pos_embed(self, x, x_jvp=None):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        x = self.pos_drop(x)
        if x_jvp is not None:
            x_jvp = self.pos_drop(x_jvp)
        return x, x_jvp


class Linear_Vision_Transformer_Last_N_Blocks(nn.Module):
    def __init__(
        self,
        last_n_blocks,
        replace_cls_token,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: str = '',
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = LinearBlock,
    ):
        super().__init__()

        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        assert block_fn == LinearBlock

        self.last_n_blocks = last_n_blocks
        self.depth = depth

        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(LinearLayerNorm, eps=1e-6)
        act_layer = act_layer or LinearGELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.replace_cls_token = replace_cls_token
        if replace_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            torch.nn.init.normal_(self.cls_token, std=1e-6)

            self.linear_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            torch.nn.init.normal_(self.cls_token, std=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.last_n_blocks > 0:
            self.blocks = LinearSequential(*[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                )
                for i in range(depth - self.last_n_blocks, depth)])
        else:
            self.blocks = None
            assert self.replace_cls_token is False

        self.norm = norm_layer(embed_dim) if not use_fc_norm else None
        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else None
        self.head_drop = nn.Dropout(drop_rate)
        assert num_classes > 0
        self.head = LinearLinear(self.embed_dim, num_classes)


    def forward(self, x, x_jvp=None):
        assert x_jvp is None

        B = x.shape[0]
        x_jvp = torch.zeros_like(x)

        if self.replace_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x[:,1:]), dim=1)

            linear_cls_tokens = self.linear_cls_token.expand(B, -1, -1)
            x_jvp = torch.cat((linear_cls_tokens, x_jvp[:,1:]), dim=1)

        if self.blocks is not None:
            x, x_jvp = self.blocks(x, x_jvp)

        if self.norm is not None:
            x, x_jvp = self.norm(x, x_jvp)

        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
            x_jvp = x_jvp[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x_jvp[:, 0]

        if self.fc_norm is not None:
            x, x_jvp = self.fc_norm(x, x_jvp)

        x = self.head_drop(x)
        x_jvp = self.head_drop(x_jvp)
        x, x_jvp = self.head(x, x_jvp)

        return x, x_jvp


    def load_pretrained_weights(self, state_dict, load_fc=False):
        if load_fc:
            classifier_weight_keys = ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']
        else:
            classifier_weight_keys = ['norm.weight', 'norm.bias']

        def adjust_block_offset(k):
            if 'blocks.' in k:
                new_num = int(k.split('.')[1]) - (self.depth - self.last_n_blocks)
                new_k = k.split('.')
                new_k[1] = str(new_num)
                return '.'.join(new_k)
            else:
                return k

        sd_new = {adjust_block_offset(k):v for k,v in state_dict.items() if ('blocks.' in k and int(k.split('.')[1]) >= self.depth - self.last_n_blocks) or k in classifier_weight_keys}
        missing, unexpected = self.load_state_dict(sd_new, strict=False)
        assert len(unexpected) == 0
        if load_fc:
            if self.replace_cls_token:
                assert all(['linear_' in m or m == 'cls_token' for m in missing])
            else:
                assert all(['linear_' in m for m in missing])
        else:
            if self.replace_cls_token:
                assert all(['linear_' in m or m.startswith('head') or m == 'cls_token' for m in missing])
            else:
                assert all(['linear_' in m or m.startswith('head') for m in missing])

        print(f"VIT Last {self.last_n_blocks} Blocks: pretrained weights loaded")

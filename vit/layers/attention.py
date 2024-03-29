# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import torch
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_Filter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_v1 = nn.Linear(dim, dim, bias=False)
        self.proj_v2 = nn.Linear(dim, dim, bias=False)
        self.complex_weight = nn.Parameter(torch.cat((torch.ones(1, 1, 1, head_dim//2 + 1, 1), torch.zeros(1, 1, 1, head_dim//2 + 1, 1)), dim=4))
        self.ac = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, N, C = x.shape  # 48, 768, 197
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, 48, 12, 197, 64
        q, k, v = qkv[0], qkv[1], qkv[2]  # 48, 12, 197, 64
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 48, 12, 197, 197
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 48, 12, 197, 197
        attn_hat = (q.transpose(1, 2).reshape(B, N, C) @ k.transpose(2, 3).reshape(B, C, N))
        attn_hat = (attn_hat + attn_hat.transpose(1, 2))/2
        attn_hat = self.ac(attn_hat)
        attn_hat_d = torch.sum(attn_hat, dim=2)
        attn_hat_d[attn_hat_d != 0] = torch.sqrt(1.0 / attn_hat_d[attn_hat_d != 0])
        Norm_attn_hat = attn_hat * attn_hat_d.unsqueeze(1) * attn_hat_d.unsqueeze(2)
        I = torch.eye(Norm_attn_hat.size(1)).to(x.device).unsqueeze(0)
        L = I - Norm_attn_hat  # 0,2
        L_2 = torch.bmm(L - I, L - I)
        out = self.proj_v1(torch.bmm(L_2 - I, v.contiguous().transpose(1, 2).reshape(B, N, C)))
        out = out - self.proj_v2(torch.bmm(L_2, v.contiguous().transpose(1, 2).reshape(B, N, C)))
        fft_v = torch.fft.rfft(v.contiguous(), dim=3, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        fft_v = fft_v * weight
        ifft_v = torch.fft.irfft(fft_v, dim=3, norm='ortho')
        x = (attn @ ifft_v).transpose(1, 2).reshape(B, N, C)  # 48, 768, 197
        x = self.proj(x)
        x = self.proj_drop(x)
        return x + out
class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

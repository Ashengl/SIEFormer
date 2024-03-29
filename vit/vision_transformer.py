# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class Attention_filter(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.K = 3  # Rank for filter
#         self.weight = nn.Parameter(torch.Tensor(1, self.num_heads, self.K, 1, 1), requires_grad=True)  # weigh for each head
#         nn.init.uniform_(self.weight)
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.relu = nn.ReLU()
#
#
#     def forward(self, x):
#         B, N, C = x.shape
#         q = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = q[0], q[1], q[2]
#
#         attn = (self.relu(q) @ self.relu(k).transpose(-2, -1)) * self.scale
#         attn = attn.reshape(B * self.num_heads, N, N)  # for convenience reshape(B, num_heads) into (B * num_heads)
#         attn_hat = (attn + attn.permute(0, 2, 1)) / 2  # attn
#
#         attn_hat_d = torch.sum(attn_hat, dim=2)
#         attn_hat_d[attn_hat_d != 0] = torch.sqrt(1.0 / attn_hat_d[attn_hat_d != 0])
#         Norm_attn_hat = attn_hat * attn_hat_d.unsqueeze(1) * attn_hat_d.unsqueeze(2)  # norm attn
#         I = torch.eye(Norm_attn_hat.size(1)).cuda().unsqueeze(0)
#         L = I - Norm_attn_hat  # laplace matrix
#
#         cheb_ = []
#         for i in range(L.shape[0]):
#             cheb = self.cheb_polynomial(L[i])
#             cheb_.append(cheb)
#         del cheb
#         cheb_ = torch.stack(cheb_, dim=0).reshape(B, self.num_heads, self.K, N, N)  # chebpolynomial for each head
#
#         v = v.reshape(B * self.num_heads, N, C // self.num_heads)
#         cheb_ = torch.sum(cheb_ * self.weight, dim=2).reshape(B * self.num_heads, N, N)  # sum for chebpoly Rank (B * num_heads, N, N)
#         """
#         cheb_ = tensor([[ 1.1925, -0.0131, -0.0120,  ..., -0.0092, -0.0108, -0.0112],
#         [-0.0131,  1.1959, -0.0164,  ..., -0.0139, -0.0118, -0.0115],
#         [-0.0120, -0.0164,  1.1956,  ..., -0.0142, -0.0123, -0.0122],
#         ...,
#         [-0.0092, -0.0139, -0.0142,  ...,  1.1902, -0.0199, -0.0204],
#         [-0.0108, -0.0118, -0.0123,  ..., -0.0199,  1.1939, -0.0179],
#         [-0.0112, -0.0115, -0.0122,  ..., -0.0204, -0.0179,  1.1947]],
#         device='cuda:0', grad_fn=<SelectBackward>)
#         """
#         attn = I - cheb_  # or attn = cheb_
#         attn = attn.softmax(dim=-1)  # or no
#         result = torch.matmul(attn, v)  # [B * num_head, N, C//num_heads]
#
#         result = result.reshape(B, self.num_heads, N, C // self.num_heads)
#         result = result.permute(0, 2, 1, 3).reshape(B, N, C)
#
#         result = self.proj(result)  # or no
#         result = self.proj_drop(result)  # or no
#
#         return result, Norm_attn_hat.reshape(B, self.num_heads, N, N)
#
#
#     def cheb_polynomial(self, laplacian):
#         """
#         Compute the Chebyshev Polynomial, according to the graph laplacian.
#
#         :param laplacian: the graph laplacian, [N, N].
#         :return: the multi order Chebyshev laplacian, [K, N, N].
#         """
#         N = laplacian.size(0)  # [N, N]
#         multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float, requires_grad=False)  # [K, N, N]
#         multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)
#
#         if self.K == 1:
#             return multi_order_laplacian
#         else:
#             multi_order_laplacian[1] = laplacian
#             if self.K == 2:
#                 return multi_order_laplacian
#             else:
#                 for k in range(2, self.K):
#                     multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1].clone()) - multi_order_laplacian[k-2].clone()
#         return multi_order_laplacian

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_vw = nn.Linear(dim, dim)
#         self.proj_va = nn.Linear(dim, dim)
#         self.proj_vb = nn.Linear(dim, dim)
#         self.complex_weight = nn.Parameter(torch.cat((torch.ones(1, 1, 1, head_dim//2 + 1, 1), torch.zeros(1, 1, 1, head_dim//2 + 1, 1)), dim=4))
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.ac = nn.ReLU()
#
#     def forward(self, x):
#         B, N, C = x.shape  # 48, 768, 197
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, 48, 12, 197, 64
#         q, k, v = qkv[0], qkv[1], qkv[2]  # 48, 12, 197, 64
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # 48, 12, 197, 197
#         attn = attn.softmax(dim=-1)
#
#         attn = self.attn_drop(attn)  # 48, 12, 197, 197
#
#         attn_hat = (q.transpose(1, 2).reshape(B, N, C) @ k.transpose(2, 3).reshape(B, C, N))
#         attn_hat = (attn_hat + attn_hat.transpose(1, 2))/2
#         attn_hat = self.ac(attn_hat)
#         attn_hat_d = torch.sum(attn_hat, dim=2)
#         attn_hat_d[attn_hat_d != 0] = torch.sqrt(1.0 / attn_hat_d[attn_hat_d != 0])
#         Norm_attn_hat = attn_hat * attn_hat_d.unsqueeze(1) * attn_hat_d.unsqueeze(2)
#         I = torch.eye(Norm_attn_hat.size(1)).cuda().unsqueeze(0)
#         L = I - Norm_attn_hat
#         L_2 = torch.bmm(L, L)
#
#         x_w = self.proj_vw(v.contiguous().transpose(1, 2).reshape(B, N, C))
#         out = self.proj_va(v.contiguous().transpose(1, 2).reshape(B, N, C) - torch.bmm(L, x_w))
#         out = out + self.proj_vb(v.contiguous().transpose(1, 2).reshape(B, N, C))
#
#         fft_v = torch.fft.rfft(v.contiguous(), dim=3, norm='ortho')
#         weight = torch.view_as_complex(self.complex_weight)
#         fft_v = fft_v * weight
#         ifft_v = torch.fft.irfft(fft_v, dim=3, norm='ortho')
#
#         x = (attn @ ifft_v).transpose(1, 2).reshape(B, N, C)  # 48, 768, 197
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x + out, attn

class Attention_Filter(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_v1 = nn.Linear(dim, dim, bias=False)
        self.proj_v2 = nn.Linear(dim, dim, bias=False)
        self.complex_weight = nn.Parameter(torch.cat((torch.ones(1, 1, 1, head_dim//2 + 1, 1), torch.zeros(1, 1, 1, head_dim//2 + 1, 1)), dim=4))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
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
        return x + out, attn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block_Filter(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_Filter(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        else:
            return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        else:
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])  # Block for basicViT  Block_filter for chebFilterViT
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class VisionTransformer_filter(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) if i < 10
            else
            Block_Filter(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])  # Block for basicViT  Block_filter for chebFilterViT
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_with_filter(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_filter(patch_size=16, **kwargs):
    model = VisionTransformer_filter(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class VisionTransformerWithLinear(nn.Module):

    def __init__(self, base_vit, num_classes=200):

        super().__init__()

        self.base_vit = base_vit
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, return_features=False):

        features = self.base_vit(x)
        features = torch.nn.functional.normalize(features, dim=-1)
        logits = self.fc(features)

        if return_features:
            return logits, features
        else:
            return logits

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.fc.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.fc.weight.copy_(w)

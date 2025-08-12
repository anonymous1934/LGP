# vit_lgp.py ─────────────────────────────────────────────────────────────
from __future__ import annotations
import inspect
from functools import partial
from typing import Optional

import torch
from torch import nn
from einops import rearrange, repeat
from pytorch_wavelets import DWTForward

from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import DropPath
try:    from timm.models import register_model
except ImportError:
    from timm.models.registry import register_model
from timm.layers import LayerNorm2d
# ────────────────────────────────────────────────
# 1. Depth‑wise Separable Conv
# ────────────────────────────────────────────────
class _DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.depth = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.point = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point(self.depth(x))

# ────────────────────────────────────────────────
# 2. AdaptiveFusionNet (通道门控 + 3×DWConv)
# ────────────────────────────────────────────────
class AdaptiveFusionNet(nn.Module):
    def __init__(self, dim: int, global_ratio: float = .33, drop_path: float = 0.):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1))      # Sigmoid→0.5
        self.conv1 = _DWConv(dim)
        self.conv2 = _DWConv(dim)
        self.conv3 = _DWConv(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_l: torch.Tensor, x_g: torch.Tensor) -> torch.Tensor:
        g = 0.5 * (torch.tanh(self.alpha) + 1.0)                 # [1,C,1,1]
        x = (1. - g) * x_l + g * x_g
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x_l + self.drop_path(x)                              # 残差 + DropPath
        x = self.conv3(x)
        return x

# ────────────────────────────────────────────────
# 3. PatchEmbed with Wavelet‑Global branch
# ────────────────────────────────────────────────
class PatchEmbedLGP(nn.Module):
    """
    局部‑全局小波 Patch Embedding
    输入:  [B, C_in, H, W]
    输出:  [B, 1 + N_patches, embed_dim]
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, *, levels=4, wavelet="haar",
                 global_ratio=.33, use_fusion_net=True,
                 dropout: float = 0.0,            # token 内部特征 dropout
                 channel_dropout: float = 0.0,    # 对多频通道使用 Dropout2D
                 token_dropout: float = 0.0,      # Patch‑Dropout 概率
                 fusion_drop_path: float = 0.0,   # FusionNet 的 DropPath
                 bias=True, **_):
        super().__init__()
        assert patch_size == 2 ** levels, \
            f"{patch_size=}, 需要满足 patch_size == 2 ** levels"

        #self.pre_norm = LayerNorm2d(in_chans)                 # LN 前
        self.post_norm = nn.LayerNorm(embed_dim)              # LN 后

        # ───────────── 1. 本地分支 ─────────────
        self.proj_local = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, patch_size,
                      stride=patch_size, bias=bias),   # 224→14 (以16为例)
            nn.GELU(),
            _DWConv(embed_dim)
        )

        # ───────────── 2. 全局多频分支 ────────
        self.dwt = DWTForward(J=1, mode="periodization", wave=wavelet)
        self.levels = levels

        self.channel_dropout = nn.Identity() if channel_dropout == 0 else nn.Dropout2d(channel_dropout)

        self.proj_global = nn.Sequential(
            nn.Conv2d(in_chans * (4 ** levels), embed_dim, 1, bias=bias),
            nn.GELU(),
            _DWConv(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=bias)
        )

        # ───────────── 3. 融合模块 ─────────────
        self.fusion = (AdaptiveFusionNet(embed_dim, global_ratio,
                                         drop_path=fusion_drop_path)
                       if use_fusion_net else None)

        # 内部正则
        #self.norm_after_fusion = nn.LayerNorm(embed_dim)
        #self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        #self.token_dropout_prob: float = token_dropout
        self.embed_dropout = nn.Identity()

        # ───────────── 4. CLS token ───────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)

        self.img_size, self.patch_size = img_size, patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.token_dropout_prob = token_dropout  # ★ 默认 0

    # ---------- 递归小波分解 ----------
    def _wp(self, x: torch.Tensor) -> torch.Tensor:
        # Store original dtype to ensure consistency
        original_dtype = x.dtype
        
        # Convert to float32 for wavelet operations to avoid precision issues
        x_float = x.float()
        feats = [x_float]
        for _ in range(self.levels):
            nxt = []
            for f in feats:
                yl, yh = self.dwt(f)
                nxt.append(yl)
                yh0 = yh[0]
                nxt.extend([yh0[:, :, i] for i in range(3)])
            feats = nxt
        result = torch.cat(feats, dim=1)
        
        # Convert back to original dtype
        return result.to(original_dtype)

    # ---------- Patch‑Dropout ----------
    def _patch_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_dropout_prob <= 0:
            return tokens
        cls_tok, patch_tok = tokens[:, :1], tokens[:, 1:]
        B, N, C = patch_tok.shape
        keep_prob = 1.0 - self.token_dropout_prob
        # 生成保留 mask
        mask = (torch.rand(B, N, device=tokens.device) < keep_prob).unsqueeze(-1)
        patch_tok = patch_tok * mask / keep_prob  # 保留值缩放保持期望
        return torch.cat([cls_tok, patch_tok], dim=1)

    # ---------- 前向 ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # ★ 预 LayerNorm
        #x = self.pre_norm(x)

        loc  = self.proj_local(x)
        glob = self.proj_global(self.channel_dropout(self._wp(x)))

        x = self.fusion(loc, glob) if self.fusion is not None else loc + glob

        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.post_norm(x)                   # ★ 后 LayerNorm
        x = self.embed_dropout(x)               # 默认恒等

        cls = repeat(self.cls_token, "() 1 d -> b 1 d", b=B)
        tokens = torch.cat([cls, x], dim=1)
        # ★ 默认不做 token dropout（可后续 schedule）
        return tokens if self.training and self.token_dropout_prob > 0 else tokens

# ────────────────────────────────────────────────
# 4. ViT backbone (无 abs pos emb)
# ────────────────────────────────────────────────
class ViT_LGP(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "pos_embed"): del self.pos_embed

    def forward_features(self, x, attn_mask=None):   # noqa
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def get_internal_loss(self):          # 兼容训练框架
        return torch.zeros([], device=self.patch_embed.cls_token.device)

# ────────────────────────────────────────────────
# 5. create_model
# ────────────────────────────────────────────────
_VT_KEYS = set(inspect.signature(VisionTransformer).parameters)
_PE_KEYS = {
    "wavelet", "levels", "use_fusion_net", "global_ratio",
    "dropout", "channel_dropout", "token_dropout", "fusion_drop_path"
}

def _create(pretrained: bool, cfg: dict, **kwargs):
    full = {**cfg, **kwargs}
    pe_kw = {k: full.pop(k) for k in list(full) if k in _PE_KEYS}
    vt_kw = {k: v for k, v in full.items() if k in _VT_KEYS}

    vt_kw.update(
        embed_layer=partial(PatchEmbedLGP, **pe_kw) if pe_kw else PatchEmbedLGP,
        class_token=False, no_embed_class=True,
        qkv_bias=vt_kw.get("qkv_bias", True),
    )
    if vt_kw.get("global_pool", "token") == "token":
        vt_kw["global_pool"] = "avg"

    model = ViT_LGP(**vt_kw)
    if pretrained:
        # load your weights here
        pass
    return model

# ────────────────────────────────────────────────
# 6. timm registry
# ────────────────────────────────────────────────
@register_model
def vit_tiny_patch16_lgp(pretrained=False, **kw):
    cfg = dict(img_size=224, patch_size=16,
               embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
               drop_path_rate=.05, global_ratio=.2)
    return _create(pretrained, cfg, **kw)

@register_model
def vit_small_patch16_lgp(pretrained=False, **kw):
    cfg = dict(img_size=224, patch_size=16,
               embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
               drop_path_rate=.1, global_ratio=.2)
    return _create(pretrained, cfg, **kw)

@register_model
def vit_base_patch16_lgp(pretrained=False, **kw):
    cfg = dict(img_size=224, patch_size=16,
               embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
               drop_path_rate=.2, global_ratio=.2)
    return _create(pretrained, cfg, **kw)

@register_model
def vit_large_patch16_lgp(pretrained=False, **kw):
    cfg = dict(img_size=224, patch_size=16,
               embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
               drop_path_rate=.3, global_ratio=.30)
    return _create(pretrained, cfg, **kw)
# ────────────────────────────────────────────────

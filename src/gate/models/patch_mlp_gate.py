"""E041e PatchMLPGate — non-convolutional *spatially structured* gate.

项目: 可见度识别研究 / 诊断协议论文证据补强（仪器效度 / 门控选择）
用途: 审稿对照——"MLP collapse 只是丢了空间结构吗?"。提供一个**保留空间结构但
不含任何卷积**的门控（MLP-Mixer 风格 token-mixing），用以区分两个假设:
  H_conv:  读出 RGB 渲染指纹需要*卷积*归纳偏置 (claim: "requires convolutional gate")
  H_space: 只需要*空间敏感*结构即可 (claim 改为 "spatially sensitive gate")
若本门也能读出 RGB 分子 (≠0) -> 支持 H_space (放宽 claim);
若本门也塌 (分子≈0 / 与平铺-MLP 无异) -> 支持 H_conv (卷积特殊)。

与 DADGGate / MLPGate / SKNetGate 相同的 I/O 契约:
    forward(x: (B, C, H, W) float) -> (B, 3) softmax  over (feature, attention, localization)

关键区别 vs gate/models/mlp_gate.py:
  MLPGate 先 adaptive_avg_pool 到 64x64 再 flatten -> 把整张图压成一个向量,
  *破坏*了 patch 间空间排列, 只剩全局统计。
  PatchMLPGate 把图切成 G×G 个 patch, 每个 patch 保留为一个 token, 用 token-mixing
  MLP 在*空间维度*上混合 (跨 patch), channel-mixing MLP 在特征维度混合 ——
  全连接但*空间结构被保留*, 且全程**无 Conv2d**。

无 nn.Conv2d / nn.Conv1d。patchify 用 reshape/unfold (无参数)。所有随机性由 trainer
在构造前设的全局 torch seed (global torch seed) 控制。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BRANCHES = 3  # feature, attention, localization


class MixerBlock(nn.Module):
    """One MLP-Mixer block: token-mixing MLP (across patches) + channel-mixing MLP.

    No convolutions. Token-mixing acts on the spatial (patch) axis so the gate
    is sensitive to *where* signal lives, unlike a pool-then-flatten MLP.
    """

    def __init__(self, n_tokens: int, dim: int, token_hidden: int, chan_hidden: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, token_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_hidden, n_tokens), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.chan_mlp = nn.Sequential(
            nn.Linear(dim, chan_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(chan_hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        # token-mixing: transpose so Linear acts over the T (spatial/patch) axis
        y = self.norm1(x).transpose(1, 2)          # (B, D, T)
        y = self.token_mlp(y).transpose(1, 2)      # (B, T, D)
        x = x + y
        x = x + self.chan_mlp(self.norm2(x))
        return x


class PatchMLPGate(nn.Module):
    """Image -> softmax weights over (feature, attention, localization).

    Pipeline (no Conv anywhere):
        x (B,C,H,W)
          -> adaptive_avg_pool to (img_size, img_size)   [param-free resize]
          -> split into grid x grid non-overlapping patches
          -> flatten each patch -> Linear patch-embed -> (B, T=grid*grid, dim)
          -> n_blocks x MixerBlock (token-mix over T + channel-mix over dim)
          -> LayerNorm -> mean over tokens -> Linear head -> 3-way softmax
    """

    N_BRANCHES = N_BRANCHES

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 64,
        grid: int = 8,
        dim: int = 128,
        n_blocks: int = 4,
        token_hidden: int = 64,
        chan_hidden: int = 256,
        dropout: float = 0.1,
        **_ignored,  # swallow MLP/conv-specific kwargs so DADGTrainer is untouched
    ) -> None:
        super().__init__()
        assert img_size % grid == 0, "img_size must be divisible by grid"
        self.in_channels = int(in_channels)
        self.img_size = int(img_size)
        self.grid = int(grid)
        self.patch = self.img_size // self.grid
        self.n_tokens = self.grid * self.grid
        self.patch_dim = self.in_channels * self.patch * self.patch
        self.embed = nn.Linear(self.patch_dim, dim)
        self.blocks = nn.ModuleList([
            MixerBlock(self.n_tokens, dim, token_hidden, chan_hidden, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, self.N_BRANCHES),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S, S) -> (B, T, patch_dim) preserving patch spatial order
        B, C, S, _ = x.shape
        g, p = self.grid, self.patch
        x = x.reshape(B, C, g, p, g, p)        # (B,C,gh,ph,gw,pw)
        x = x.permute(0, 2, 4, 1, 3, 5)        # (B,gh,gw,C,ph,pw)
        x = x.reshape(B, g * g, C * p * p)     # (B, T, patch_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = F.adaptive_avg_pool2d(x, self.img_size)   # param-free resize
        t = self._patchify(x)                          # (B, T, patch_dim)
        h = self.embed(t)                              # (B, T, dim)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).mean(dim=1)                   # (B, dim) global token pool
        return F.softmax(self.head(h), dim=-1)

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_patch_mlp_gate(**kwargs) -> PatchMLPGate:
    """Factory for config-driven construction."""
    return PatchMLPGate(**kwargs)

"""DADG (Divergence-Aware Dynamic Gate) network.

Lightweight image-conditional gate that outputs a 3-way softmax over KD
branches (feature, attention, localization). Trained against divergence-
derived soft targets with stop_gradient from the downstream KD loss to
prevent the collapse observed in prior gate variants.

See: gate/TODO_DADG.md, plan file zesty-hopping-rocket.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DADGGate(nn.Module):
    """Image → softmax weights over (feature, attention, localization).

    Architecture:
        Conv-BN-ReLU × 5 (channels 32→64→128→128→256, stride-2 each)
        AdaptiveAvgPool → 256-d vector
        MLP 256→128→3 (softmax)

    Parameter count ≈ 0.7M.
    """

    N_BRANCHES = 3  # feature, attention, localization

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple[int, ...] = (32, 64, 128, 128, 256),
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_prev = in_channels
        for c in widths:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_prev, c, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                )
            )
            c_prev = c
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c_prev, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, self.N_BRANCHES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) foggy image. Returns (B, 3) softmax weights."""
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        logits = self.head(h)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_dadg(**kwargs) -> DADGGate:
    """Factory for config-driven construction."""
    return DADGGate(**kwargs)


# ---------------------------------------------------------------------------
# E041 gate-architecture factory
# 项目: 可见度识别研究 / 诊断框架论文证据补强 wave1
# ---------------------------------------------------------------------------
GATE_ARCHS = ("conv", "mlp", "sknet")


def build_gate(arch: str = "conv", **kwargs) -> nn.Module:
    """Build a gate network by architecture name.

    Default "conv" returns the original DADGGate with identical kwargs, so all
    existing call sites and checkpoints are unaffected (E041 hard requirement).
    "mlp" returns the parameter-aligned MLPGate (gate/models/mlp_gate.py).
    """
    arch = (str(arch or "conv")).strip().lower()
    if arch in ("conv", "dadg", ""):
        return DADGGate(**kwargs)
    if arch == "mlp":
        from gate.models.mlp_gate import MLPGate  # local import: no hard dependency
        return MLPGate(**kwargs)
    if arch == "sknet":
        from gate.models.sknet_gate import SKNetGate  # local import: no hard dependency
        return SKNetGate(**kwargs)
    raise ValueError(f"Unknown gate arch {arch!r}; expected one of {GATE_ARCHS}")

"""Physical prior extraction for TSP-conditioned KD.

All functions accept RGB tensors in [0, 1] with shape (B, 3, H, W). The main
paper TSP is `tsp_grad_mag`: hard-min DCP transmission followed by 3x3 Sobel
gradient magnitude. Later probes add non-RGB-guided candidate definitions that
try to control hard-min artifacts without using guided filtering or soft
matting. These candidate names are internal experiment names; the manuscript
should call only the finally selected definition "TSP".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_odd(k: int) -> int:
    return int(k) if int(k) % 2 == 1 else int(k) + 1


def compute_dark_channel(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    min_c = img.min(dim=1, keepdim=True).values
    pad = patch_size // 2
    return -F.max_pool2d(-min_c, kernel_size=patch_size, stride=1, padding=pad)


def soft_dark_channel(img: torch.Tensor, patch_size: int = 15, tau: float = 0.04) -> torch.Tensor:
    """Soft-min dark channel without RGB-guided refinement.

    This uses avg-pooling in exp space as a memory-friendly approximation to a
    local soft minimum. It reduces hard source-pixel switching while keeping the
    extraction non-RGB-guided.
    """
    min_c = img.min(dim=1, keepdim=True).values
    pad = patch_size // 2
    z = torch.exp((-min_c / tau).clamp(max=30.0))
    pooled = F.avg_pool2d(z, kernel_size=patch_size, stride=1, padding=pad)
    return -tau * torch.log(pooled.clamp_min(1e-12))


def percentile_dark_channel(img: torch.Tensor, patch_size: int = 15, q: float = 0.05) -> torch.Tensor:
    """Low-percentile dark channel without RGB-guided refinement.

    For patch size 15, q=0.05 and q=0.10 correspond roughly to the 12th and
    23rd smallest local channel-min values. This reduces sensitivity to a
    single darkest pixel while preserving the dark-channel construction.
    """
    min_c = img.min(dim=1, keepdim=True).values
    pad = patch_size // 2
    padded = F.pad(min_c, (pad, pad, pad, pad), mode="reflect")
    patches = F.unfold(padded, kernel_size=patch_size)
    patches = patches.view(img.size(0), 1, patch_size * patch_size, img.size(2), img.size(3))
    kth = max(1, min(patch_size * patch_size, int(torch.ceil(torch.tensor(q * patch_size * patch_size)).item())))
    return patches.kthvalue(kth, dim=2).values


def estimate_atmospheric_light(img: torch.Tensor, dark: torch.Tensor) -> torch.Tensor:
    b = img.size(0)
    flat_dark = dark.view(b, -1)
    k = max(1, int(0.001 * flat_dark.size(-1)))
    _, topk_idx = flat_dark.topk(k, dim=-1)
    flat_img = img.view(b, 3, -1)
    topk_idx_3ch = topk_idx.unsqueeze(1).expand(-1, 3, -1)
    return flat_img.gather(2, topk_idx_3ch).mean(dim=-1).clamp(1e-3, 1.0)


def estimate_transmission_dcp(
    img: torch.Tensor,
    omega: float = 0.95,
    patch_size: int = 15,
    t_min: float = 0.05,
    pool: str = "hardmin",
    percentile_q: float = 0.05,
) -> torch.Tensor:
    if pool == "softmin":
        dark = soft_dark_channel(img, patch_size=patch_size)
    else:
        dark = compute_dark_channel(img, patch_size=patch_size)
    a = estimate_atmospheric_light(img, dark).view(img.size(0), 3, 1, 1)
    normalized = img / a.clamp_min(1e-3)
    if pool == "softmin":
        dark_norm = soft_dark_channel(normalized.clamp(0.0, 5.0), patch_size=patch_size)
    elif pool == "percentile":
        dark_norm = percentile_dark_channel(
            normalized.clamp(0.0, 5.0), patch_size=patch_size, q=percentile_q
        )
    else:
        dark_norm = compute_dark_channel(normalized.clamp(0.0, 5.0), patch_size=patch_size)
    return (1.0 - omega * dark_norm).clamp(t_min, 1.0)


def _sobel_kernels(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)
    return sx.view(1, 1, 3, 3), sy.view(1, 1, 3, 3)


def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    sx, sy = _sobel_kernels(x.dtype, x.device)
    gx = F.conv2d(x, sx, padding=1)
    gy = F.conv2d(x, sy, padding=1)
    return (gx.square() + gy.square()).sqrt()


def _sobel_mag_channels(x: torch.Tensor) -> torch.Tensor:
    sx, sy = _sobel_kernels(x.dtype, x.device)
    channels = x.size(1)
    sx = sx.repeat(channels, 1, 1, 1)
    sy = sy.repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, sx, padding=1, groups=channels)
    gy = F.conv2d(x, sy, padding=1, groups=channels)
    return (gx.square() + gy.square()).sum(dim=1, keepdim=True).sqrt()


def _dog_mag(x: torch.Tensor) -> torch.Tensor:
    """Derivative-of-Gaussian magnitude with sigma approximately 1."""
    smooth = torch.tensor([1, 4, 6, 4, 1], dtype=x.dtype, device=x.device) / 16.0
    deriv = torch.tensor([-1, -2, 0, 2, 1], dtype=x.dtype, device=x.device) / 8.0
    dx = deriv.view(1, 1, 1, 5)
    dy = deriv.view(1, 1, 5, 1)
    sx = smooth.view(1, 1, 1, 5)
    sy = smooth.view(1, 1, 5, 1)
    gx = F.conv2d(F.pad(F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), sy), (2, 2, 0, 0), mode="reflect"), dx)
    gy = F.conv2d(F.pad(F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), sx), (0, 0, 2, 2), mode="reflect"), dy)
    return (gx.square() + gy.square()).sqrt()


def _norm01_per_image(x: torch.Tensor) -> torch.Tensor:
    b = x.size(0)
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(b, 1, 1, 1)
    hi = flat.max(dim=1).values.view(b, 1, 1, 1)
    return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _norm_channels_per_image(x: torch.Tensor) -> torch.Tensor:
    """Normalize each channel independently within each image."""
    b, c = x.shape[:2]
    flat = x.flatten(2)
    lo = flat.min(dim=2).values.view(b, c, 1, 1)
    hi = flat.max(dim=2).values.view(b, c, 1, 1)
    return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def rgb_luma(img: torch.Tensor) -> torch.Tensor:
    coeff = img.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (img * coeff).sum(dim=1, keepdim=True).clamp(0.0, 1.0)


def rgb_gray(img: torch.Tensor) -> torch.Tensor:
    """One-channel RGB appearance control.

    This keeps ordinary RGB luminance while removing color-channel capacity.
    It is a reviewer-control condition, not a physical prior.
    """
    return rgb_luma(img)


def rgb_sobel_mag(img: torch.Tensor) -> torch.Tensor:
    """Sobel edge-magnitude control extracted directly from RGB luminance."""
    return _sobel_mag(rgb_luma(img))


def rgb_equalized_gray(img: torch.Tensor) -> torch.Tensor:
    """Per-image standardized RGB-luminance control.

    This removes most global intensity/dynamic-range differences while
    preserving appearance texture. It tests whether RGB shortcut reduction can
    be explained by simple intensity equalization rather than physical
    degradation structure.
    """
    gray = rgb_luma(img)
    mean = gray.mean(dim=(2, 3), keepdim=True)
    std = gray.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return torch.sigmoid((gray - mean) / std)


def _gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    kernel_1d = torch.tensor([1, 4, 6, 4, 1], dtype=x.dtype, device=x.device) / 16.0
    kx = kernel_1d.view(1, 1, 1, 5)
    ky = kernel_1d.view(1, 1, 5, 1)
    x = F.pad(x, (2, 2, 0, 0), mode="reflect")
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, 2, 2), mode="reflect")
    return F.conv2d(x, ky)


def _median_blur(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    k = _ensure_odd(k)
    pad = k // 2
    patches = F.unfold(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=k)
    patches = patches.view(x.size(0), 1, k * k, x.size(2), x.size(3))
    return patches.median(dim=2).values


def tsp_rank_fast(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size)
    b, _, h, w = t.shape
    flat = t.view(b, -1)
    ranks = flat.argsort(dim=-1).argsort(dim=-1).to(t.dtype) / max(1.0, float(h * w - 1))
    return ranks.view(b, 1, h, w)


def tsp_gradient_direction(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size)
    sx, sy = _sobel_kernels(t.dtype, t.device)
    return torch.atan2(F.conv2d(t, sy, padding=1), F.conv2d(t, sx, padding=1))


def tsp_gradient_magnitude(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size)
    return _sobel_mag(t)


def tsp_hardmin_gaussian_grad(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size)
    return _sobel_mag(_gaussian_blur(t))


def tsp_hardmin_median_grad(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size)
    return _sobel_mag(_median_blur(t))


def tsp_hardmin_log_grad(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size)
    return _sobel_mag(torch.log(t.clamp_min(1e-3)))


def tsp_multiscale_hardmin_grad(img: torch.Tensor) -> torch.Tensor:
    maps = []
    for patch in (7, 15, 31):
        maps.append(_norm01_per_image(tsp_gradient_magnitude(img, patch_size=patch)))
    return torch.stack(maps, dim=0).mean(dim=0)


def tsp_softmin_log_grad(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size, pool="softmin")
    return _sobel_mag(torch.log(t.clamp_min(1e-3)))


def tsp_hardmin_dog_grad(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size)
    return _dog_mag(t)


def tsp_percentile05_dog_grad(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size, pool="percentile", percentile_q=0.05)
    return _dog_mag(t)


def tsp_percentile10_dog_grad(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    t = estimate_transmission_dcp(img, patch_size=patch_size, pool="percentile", percentile_q=0.10)
    return _dog_mag(t)


def tsp_source_switch_soft_grad(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    # Training-time approximation: use local dark-channel gradient as a
    # differentiable proxy for switch density. E021 uses the exact argmin-based
    # source-switch diagnostic; this registered candidate is for E022 only if
    # E021 selects the suppression family.
    t = estimate_transmission_dcp(img, patch_size=patch_size)
    base = _sobel_mag(t)
    dark = compute_dark_channel(img, patch_size=patch_size)
    density = F.avg_pool2d((_sobel_mag(dark) > 0.02).to(img.dtype), kernel_size=5, stride=1, padding=2)
    return base / (1.0 + 2.0 * density)


def raw_transmission(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    return estimate_transmission_dcp(img, patch_size=patch_size)


def raw_dark_channel(img: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    return compute_dark_channel(img, patch_size=patch_size)


def airlight_angular_grad(
    img: torch.Tensor,
    patch_size: int = 15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gradient magnitude of airlight-centered color direction `u`.

    This is the F046 `grad_u` candidate. It replaces the DCP transmission
    gradient as a gate condition with an angular/color-direction gradient:
        c = I - A, r = ||c||, u = c / r.
    The output is one normalized map per image, matching the one-channel TSP
    gate-input interface.
    """
    dark = compute_dark_channel(img, patch_size=patch_size)
    airlight = estimate_atmospheric_light(img, dark).view(img.size(0), 3, 1, 1)
    centered = img - airlight
    radius = centered.square().sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    direction = centered / radius
    return _norm01_per_image(_sobel_mag_channels(direction))


def airlight_radial_grad(
    img: torch.Tensor,
    patch_size: int = 15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gradient magnitude of log airlight-centered color radius.

    Under the haze model, ||I-A|| contracts with transmission. F046 showed
    this radial contraction is a real mechanism, but not a standalone object
    boundary cue. Here it is exposed only as one KD condition channel.
    """
    dark = compute_dark_channel(img, patch_size=patch_size)
    airlight = estimate_atmospheric_light(img, dark).view(img.size(0), 3, 1, 1)
    centered = img - airlight
    radius = centered.square().sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    return _norm01_per_image(_sobel_mag(torch.log(radius.clamp_min(eps))))


def dcp_source_switch_artifact(
    img: torch.Tensor,
    patch_size: int = 15,
) -> torch.Tensor:
    """Training-time proxy for DCP hard-min source-switch/artifact risk.

    F044/F046 showed current TSP is strongly tied to dark-channel source-switch
    mechanics. Exact argmin-switch maps are too expensive for online training,
    so this uses local dark-channel gradient density as a bounded artifact cue.
    """
    dark = compute_dark_channel(img, patch_size=patch_size)
    switch_like = (_sobel_mag(dark) > 0.02).to(img.dtype)
    density = F.avg_pool2d(switch_like, kernel_size=5, stride=1, padding=2)
    return density.clamp(0.0, 1.0)


def artifact_suppressed_tsp_grad(
    img: torch.Tensor,
    patch_size: int = 15,
) -> torch.Tensor:
    """TSP gradient with online DCP source-switch risk suppressed.

    F044/F046 indicate that high current TSP often follows DCP hard-min
    source-switch mechanics rather than object-aligned visibility structure.
    This candidate removes that artifact component before the DADG gate sees
    the condition, instead of appending the artifact cue and expecting the gate
    to learn the suppression from a small KD objective.
    """
    tsp = _norm01_per_image(tsp_gradient_magnitude(img, patch_size=patch_size))
    artifact = dcp_source_switch_artifact(img, patch_size=patch_size)
    return _norm01_per_image(tsp * (1.0 - artifact).clamp(0.0, 1.0))


def physics_multicue(
    img: torch.Tensor,
    patch_size: int = 15,
) -> torch.Tensor:
    """Four-channel physical condition for mechanism-guided KD.

    Channels:
    1. DCP transmission-gradient TSP (P2 main condition).
    2. Airlight-centered angular gradient (`grad_u`, F046/F047).
    3. Airlight-centered radial/log-radius gradient.
    4. DCP source-switch/artifact-risk proxy.

    The cue is intended as a KD condition/reliability signal, not as an RGB
    replacement for recognition.
    """
    tsp = _norm01_per_image(tsp_gradient_magnitude(img, patch_size=patch_size))
    angular = airlight_angular_grad(img, patch_size=patch_size)
    radial = airlight_radial_grad(img, patch_size=patch_size)
    artifact = dcp_source_switch_artifact(img, patch_size=patch_size)
    return _norm_channels_per_image(torch.cat([tsp, angular, radial, artifact], dim=1))


PRIOR_REGISTRY = {
    "rgb_gray": rgb_gray,
    "rgb_sobel_mag": rgb_sobel_mag,
    "rgb_equalized_gray": rgb_equalized_gray,
    "tsp_rank": tsp_rank_fast,
    "tsp_grad_dir": tsp_gradient_direction,
    "tsp_grad_mag": tsp_gradient_magnitude,
    "current_hardmin_grad": tsp_gradient_magnitude,
    "hardmin_gaussian_grad": tsp_hardmin_gaussian_grad,
    "hardmin_median_grad": tsp_hardmin_median_grad,
    "hardmin_log_grad": tsp_hardmin_log_grad,
    "multiscale_hardmin_grad": tsp_multiscale_hardmin_grad,
    "softmin_log_grad": tsp_softmin_log_grad,
    "hardmin_dog_grad": tsp_hardmin_dog_grad,
    "percentile05_dog_grad": tsp_percentile05_dog_grad,
    "percentile10_dog_grad": tsp_percentile10_dog_grad,
    "source_switch_soft_grad": tsp_source_switch_soft_grad,
    "airlight_angular_grad": airlight_angular_grad,
    "airlight_radial_grad": airlight_radial_grad,
    "dcp_source_switch_artifact": dcp_source_switch_artifact,
    "artifact_suppressed_tsp_grad": artifact_suppressed_tsp_grad,
    "physics_multicue": physics_multicue,
    "raw_transmission": raw_transmission,
    "raw_dark_channel": raw_dark_channel,
}


def build_gate_input(
    img: torch.Tensor,
    mode: str = "rgb",
    use_dark_channel: bool = False,
) -> torch.Tensor:
    if use_dark_channel:
        return torch.cat([img, compute_dark_channel(img)], dim=1)
    if mode == "rgb":
        return img
    if mode == "rgb_plus_tsp_rank":
        return torch.cat([img, tsp_rank_fast(img)], dim=1)
    if mode == "rgb_plus_tsp_grad_dir":
        return torch.cat([img, tsp_gradient_direction(img)], dim=1)
    if mode == "rgb_plus_tsp_grad_mag":
        return torch.cat([img, tsp_gradient_magnitude(img)], dim=1)
    if mode == "rgb_plus_artifact_suppressed_tsp_grad":
        return torch.cat([img, artifact_suppressed_tsp_grad(img)], dim=1)
    if mode == "rgb_plus_physics_multicue":
        return torch.cat([img, physics_multicue(img)], dim=1)
    try:
        return PRIOR_REGISTRY[mode](img)
    except KeyError as exc:
        raise ValueError(f"Unknown gate input mode: {mode}") from exc

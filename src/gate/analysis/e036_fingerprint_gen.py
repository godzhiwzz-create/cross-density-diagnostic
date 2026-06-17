"""E036 fingerprint generator: deterministic shortcut spike-in for DI calibration.

项目: 可见度识别研究 / 诊断框架论文证据补强 wave1（与 p3_method/p3_residual 无关）

Generates "spiked" copies of the synthetic Cityscapes-Foggy splits by adding one
of three controlled, known shortcuts ("fingerprints") at a graded dose. The DI
dose-response probe (e036_di_dose_response.py) then measures whether a trained
DADG gate's real-vs-synthetic separation moves monotonically with dose. E037
reuses the SAME generator (--full) to build a spiked TRAINING set.

Three fingerprint families (each: zero-dose + 4 graded doses):
  colorbias  : LAB-space a/b micro-shift (constant chroma offset). Pure
               appearance shift, no fog physics. Pre-registered DI sign: +.
  noise      : fixed-kernel Gaussian blur of a Gaussian+Poisson noise field,
               added as a stable per-image noise signature. Pre-registered: +.
  jpeg       : JPEG quality-factor ladder (re-encode at decreasing quality).
               JPEG block/ringing artifacts partly MIMIC compression already in
               RTTS. Pre-registered DI sign: -/mixed (may REDUCE real-vs-synth
               gap as synthetic acquires RTTS-like compression structure).

Determinism contract
--------------------
* The per-dose fingerprint CONFIG (delta a/b, noise sigma/lambda, JPEG quality)
  is a fixed ladder, identical across the three beta densities. This is
  load-bearing: the DI denominator is the synthetic CROSS-density MMD; if a
  family/dose were applied with different params per beta, the spike itself
  would open a cross-density gap and contaminate the denominator (the
  cross-concentration protection).
* The per-image stochastic component (the noise family's random field) is seeded
  by sha256(base_seed | family | dose_idx | image_stem) so every rerun is
  byte-reproducible and independent of file ordering.
* colorbias and jpeg are fully deterministic functions of the input image.

Probe-subset vs full mode
-------------------------
Default ("probe" mode) writes only a capped subset per (beta, dose) (<= --max-probe,
default 500) -- enough for the DI probe, cheap on disk. --full re-spikes every
image in the chosen split (for E037 training). Zero-dose is materialized too so
the probe has a matched within-pipeline baseline (same resize/codec round-trip).

CPU-only. No GPU, no training, no model load.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Remote-vs-local path roots. This file is authored locally but RUNS on dl-host;
# keep the dl-host layout as the default and allow --data-root / --out-root
# overrides for smoke checks.
# ----------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path(
    "./数据集_prepared/low_visibility_kd"
)
DEFAULT_OUT_ROOT = Path(
    "./gate/experiments_dlhost"
    "/e036_fingerprint_dose_response/spiked"
)

# Synthetic beta splits, in the SAME order/tags used by di_signal_ladder_probe.py.
BETA_SPLITS = {
    "beta_0.005": "cityscapes_yolo/foggy_beta_0_005",
    "beta_0.01": "cityscapes_yolo/foggy_beta_0_01",
    "beta_0.02": "cityscapes_yolo/foggy_beta_0_02",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ----------------------------------------------------------------------------
# Pre-registered dose ladders (dose index 0 == zero dose == identity-through-
# pipeline). Identical across all three beta densities by construction.
# ----------------------------------------------------------------------------
# colorbias: constant (da, db) offset in CIELAB (OpenCV-free, manual sRGB<->LAB).
COLORBIAS_DOSES = [
    (0.0, 0.0),    # dose 0: zero
    (2.0, 2.0),    # dose 1
    (5.0, 5.0),    # dose 2
    (10.0, 10.0),  # dose 3
    (16.0, 16.0),  # dose 4
]
# noise: (gaussian_sigma_8bit, poisson_lambda_scale). The noise field is blurred
# by a fixed 3x3 Gaussian kernel to give a stable spatial "signature".
NOISE_DOSES = [
    (0.0, 0.0),   # dose 0: zero
    (2.0, 0.25),  # dose 1
    (4.0, 0.5),   # dose 2
    (8.0, 1.0),   # dose 3
    (14.0, 2.0),  # dose 4
]
# jpeg: quality factor. dose 0 sentinel (-1) => NO re-encode (pass-through). For
# doses 1..4 the image is JPEG-encoded at decreasing quality.
JPEG_DOSES = [-1, 85, 60, 40, 20]

DOSE_LADDERS = {
    "colorbias": COLORBIAS_DOSES,
    "noise": NOISE_DOSES,
    "jpeg": JPEG_DOSES,
}
N_DOSES = 5  # zero + 4

# Fixed 3x3 Gaussian kernel for the noise family's spatial signature.
_GAUSS3 = np.array(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64
)
_GAUSS3 /= _GAUSS3.sum()


def _conv3_reflect(field: np.ndarray) -> np.ndarray:
    """Depthwise 3x3 convolution with reflect padding (numpy, per HxWxC)."""
    padded = np.pad(field, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    out = np.zeros_like(field, dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            out += _GAUSS3[di, dj] * padded[di : di + field.shape[0], dj : dj + field.shape[1], :]
    return out


# ----------------------------------------------------------------------------
# Manual sRGB <-> CIELAB (D65) so the script has zero OpenCV/skimage dependency
# and is byte-identical across environments.
# ----------------------------------------------------------------------------
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


_M_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]]
)
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])  # D65


def _f_lab(t: np.ndarray) -> np.ndarray:
    d = 6.0 / 29.0
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    d = 6.0 / 29.0
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1] HxWx3 -> Lab (L in [0,100], a/b roughly [-128,127])."""
    lin = _srgb_to_linear(rgb)
    xyz = lin @ _M_RGB2XYZ.T
    xyz = xyz / _WHITE
    f = _f_lab(xyz)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Lab -> rgb in [0,1] HxWx3 (clamped)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz = np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], axis=-1) * _WHITE
    lin = xyz @ _M_XYZ2RGB.T
    rgb = _linear_to_srgb(lin)
    return np.clip(rgb, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Fingerprint application (each returns a uint8 HxWx3 RGB array).
# ----------------------------------------------------------------------------
def apply_colorbias(img_u8: np.ndarray, dose_idx: int) -> np.ndarray:
    da, db = COLORBIAS_DOSES[dose_idx]
    if da == 0.0 and db == 0.0:
        return img_u8
    rgb = img_u8.astype(np.float64) / 255.0
    lab = rgb_to_lab(rgb)
    lab[..., 1] += da
    lab[..., 2] += db
    out = lab_to_rgb(lab)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def apply_noise(img_u8: np.ndarray, dose_idx: int, seed: int) -> np.ndarray:
    sigma, lam_scale = NOISE_DOSES[dose_idx]
    if sigma == 0.0 and lam_scale == 0.0:
        return img_u8
    rng = np.random.default_rng(seed)
    h, w, c = img_u8.shape
    base = img_u8.astype(np.float64)
    # Gaussian component (signal-independent) + Poisson-like component
    # (signal-dependent shot noise), both blurred by the fixed kernel so the
    # "signature" has stable spatial statistics across images.
    gauss = rng.standard_normal((h, w, c)) * sigma
    if lam_scale > 0.0:
        # shot noise: variance proportional to intensity; centered.
        lam = np.clip(base * lam_scale / 255.0 * 12.0, 1e-3, None)
        poisson = rng.poisson(lam).astype(np.float64) - lam
    else:
        poisson = np.zeros_like(base)
    field = _conv3_reflect(gauss + poisson)
    out = base + field
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def apply_jpeg(img_u8: np.ndarray, dose_idx: int) -> np.ndarray:
    quality = JPEG_DOSES[dose_idx]
    if quality < 0:
        return img_u8
    buf = io.BytesIO()
    Image.fromarray(img_u8, mode="RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def apply_fingerprint(img_u8: np.ndarray, family: str, dose_idx: int, seed: int) -> np.ndarray:
    if family == "colorbias":
        return apply_colorbias(img_u8, dose_idx)
    if family == "noise":
        return apply_noise(img_u8, dose_idx, seed)
    if family == "jpeg":
        return apply_jpeg(img_u8, dose_idx)
    raise ValueError(f"Unknown family: {family}")


# ----------------------------------------------------------------------------
# Determinism helpers
# ----------------------------------------------------------------------------
def image_seed(base_seed: int, family: str, dose_idx: int, stem: str) -> int:
    key = f"{base_seed}|{family}|{dose_idx}|{stem}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 31)


def list_split_images(data_root: Path, split_rel: str, split: str) -> list[Path]:
    img_dir = data_root / split_rel / "images" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing image dir: {img_dir}")
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def subset_paths(paths: list[Path], max_n: int, base_seed: int, beta_tag: str) -> list[Path]:
    """Deterministic capped subset (matched across doses via the same draw)."""
    if max_n <= 0 or len(paths) <= max_n:
        return paths
    rng = np.random.default_rng(int.from_bytes(
        hashlib.sha256(f"{base_seed}|{beta_tag}".encode()).digest()[:8], "big") % (2 ** 31))
    idx = sorted(rng.choice(len(paths), size=max_n, replace=False).tolist())
    return [paths[i] for i in idx]


# ----------------------------------------------------------------------------
# Main generation loop
def _spike_one(task: tuple) -> int:
    """Parallel worker: load -> fingerprint -> save one image (picklable)."""
    src_str, dst_str, family, dose_idx, seed = task
    img = np.asarray(Image.open(src_str).convert("RGB"), dtype=np.uint8)
    out = apply_fingerprint(img, family, dose_idx, seed)
    Image.fromarray(out, mode="RGB").save(dst_str, format="PNG")
    return 1


# ----------------------------------------------------------------------------
def generate(
    *,
    families: Iterable[str],
    data_root: Path,
    out_root: Path,
    split: str,
    betas: list[str],
    base_seed: int,
    max_probe: int,
    full: bool,
    limit_images: int,
    overwrite: bool,
    doses: list[int] | None = None,
    workers: int = 1,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "base_seed": base_seed,
        "split": split,
        "full": full,
        "max_probe": None if full else max_probe,
        "families": {},
        "dose_ladders": {
            "colorbias": [list(x) for x in COLORBIAS_DOSES],
            "noise": [list(x) for x in NOISE_DOSES],
            "jpeg": JPEG_DOSES,
        },
        "preregistered_di_sign": {"colorbias": "+", "noise": "+", "jpeg": "-/mixed"},
        "outputs": [],
    }

    for beta_tag in betas:
        split_rel = BETA_SPLITS[beta_tag]
        all_paths = list_split_images(data_root, split_rel, split)
        if limit_images > 0:
            all_paths = all_paths[:limit_images]
        sel = all_paths if full else subset_paths(all_paths, max_probe, base_seed, beta_tag)

        for family in families:
            fam_counts = manifest["families"].setdefault(family, {})
            for dose_idx in (doses if doses else range(N_DOSES)):
                dst_dir = out_root / family / f"dose{dose_idx}" / beta_tag / "images" / split
                dst_dir.mkdir(parents=True, exist_ok=True)
                written = 0
                tasks = []
                for src in sel:
                    dst = dst_dir / (src.stem + ".png")
                    if dst.exists() and not overwrite:
                        written += 1
                        continue
                    tasks.append((str(src), str(dst), family, dose_idx,
                                  image_seed(base_seed, family, dose_idx, src.stem)))
                # write PNG (lossless) so the only lossy step is the JPEG
                # fingerprint itself; everything else is bit-stable.
                if tasks:
                    if workers > 1:
                        with ProcessPoolExecutor(max_workers=workers) as ex:
                            for _ in ex.map(_spike_one, tasks, chunksize=16):
                                written += 1
                    else:
                        for t in tasks:
                            written += _spike_one(t)
                key = f"{beta_tag}/dose{dose_idx}"
                fam_counts[key] = written
                manifest["outputs"].append({
                    "family": family, "beta": beta_tag, "dose_idx": dose_idx,
                    "dir": str(dst_dir), "n_images": written,
                })
                print(f"[gen] {family} {beta_tag} dose{dose_idx}: {written} imgs -> {dst_dir}",
                      flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="E036 deterministic fingerprint spike-in generator")
    ap.add_argument("--families", nargs="+", default=["colorbias", "noise", "jpeg"],
                    choices=["colorbias", "noise", "jpeg"])
    ap.add_argument("--betas", nargs="+", default=list(BETA_SPLITS), choices=list(BETA_SPLITS))
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="val for the DI probe (E036); train for the E037 spiked training set.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--base-seed", type=int, default=20260611)
    ap.add_argument("--max-probe", type=int, default=500,
                    help="Cap per (beta,dose) in probe mode. <=0 means all.")
    ap.add_argument("--full", action="store_true",
                    help="Spike every image in the split (E037 training). Ignores --max-probe.")
    ap.add_argument("--limit-images", type=int, default=0, help="smoke: only first N source imgs")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--doses", type=int, nargs="+", default=None,
                    help="Subset of dose indices 0-4; default all.")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--manifest-name", default="e036_fingerprint_manifest.json")
    args = ap.parse_args()

    manifest = generate(
        families=args.families,
        data_root=args.data_root,
        out_root=args.out_root,
        split=args.split,
        betas=args.betas,
        base_seed=args.base_seed,
        max_probe=args.max_probe,
        full=args.full,
        limit_images=args.limit_images,
        overwrite=args.overwrite,
        doses=args.doses,
        workers=args.workers,
    )
    man_path = args.out_root / args.manifest_name
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] {man_path}")
    print(json.dumps({"families": list(manifest["families"]), "n_outputs": len(manifest["outputs"]),
                      "out_root": str(args.out_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

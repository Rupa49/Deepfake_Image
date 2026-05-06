"""
forensics.py  –  Multi-Signal Forensic Analysis Module
Provides three independent deepfake indicators beyond the CNN:
  1. ELA  (Error Level Analysis)       – detects double-compression artifacts
  2. Noise-Map Analysis                – detects inconsistent sensor noise patterns
  3. Image Metadata Inspection         – flags suspicious EXIF / PIL metadata
"""
import io
import numpy as np
import cv2
from PIL import Image, ExifTags
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Error Level Analysis (ELA)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ela(pil_img: Image.Image, quality: int = 90) -> tuple[np.ndarray, float]:
    """
    Re-compress image at `quality` and compute pixel-wise absolute difference.
    Returns:
        ela_img  : (H, W, 3) uint8 amplified difference image
        ela_score: float  0..1 (higher = more suspicious)
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_arr = np.array(pil_img.convert("RGB")).astype(np.float32)
    reco_arr = np.array(recompressed).astype(np.float32)

    diff = np.abs(orig_arr - reco_arr)
    # Amplify for visibility
    ela_img = np.clip(diff * 15, 0, 255).astype(np.uint8)

    # Score: fraction of highly-different pixels (threshold at 20 after amplification)
    ela_score = float(np.mean(ela_img > 20) )
    return ela_img, ela_score


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Noise-Map Consistency Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_noise_map(pil_img: Image.Image) -> tuple[np.ndarray, float]:
    """
    Estimate sensor noise via Laplacian high-pass filter.
    Genuine camera images have consistent noise; composited / GAN images
    often show patchwork noise patterns.
    Returns:
        noise_img   : (H, W, 3) uint8 colour-mapped noise image
        inconsistency: float  0..1 (higher = more suspicious)
    """
    gray = np.array(pil_img.convert("L")).astype(np.float32)
    h, w = gray.shape

    # Laplacian noise residual
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise   = gray - blurred

    # Divide image into 8×8 blocks and measure std-dev variance
    block = 8
    h_b, w_b = h // block, w // block
    stds = []
    for r in range(h_b):
        for c in range(w_b):
            patch = noise[r*block:(r+1)*block, c*block:(c+1)*block]
            stds.append(float(np.std(patch)))

    stds = np.array(stds)
    # Coefficient of variation of block-level noise std-devs
    inconsistency = float(np.std(stds) / (np.mean(stds) + 1e-8))
    # Clip to [0,1] range (>1.0 is very inconsistent)
    inconsistency = min(inconsistency, 1.0)

    # Visualise: normalise noise residual and apply heat-map
    norm = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    noise_heat = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_INFERNO)
    noise_img  = cv2.cvtColor(noise_heat, cv2.COLOR_BGR2RGB)
    return noise_img, inconsistency


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Metadata Inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_metadata(pil_img: Image.Image) -> dict:
    """
    Extract and flag suspicious metadata.
    Returns dict with keys:
        has_exif     : bool
        software     : str | None
        flags        : list[str]  – human-readable warning strings
        suspicion    : float 0..1
    """
    flags      = []
    suspicion  = 0.0
    software   = None
    has_exif   = False

    exif_data = pil_img._getexif() if hasattr(pil_img, "_getexif") else None
    if exif_data:
        has_exif = True
        tag_map  = {v: k for k, v in ExifTags.TAGS.items()}
        readable = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}

        software = readable.get("Software", None)
        if software:
            ai_keywords = ["stable diffusion", "midjourney", "dall", "gan", "deepfake",
                           "faceswap", "roop", "insightface", "diffusion", "generated"]
            if any(kw in str(software).lower() for kw in ai_keywords):
                flags.append(f"⚠ Software tag suggests AI generation: '{software}'")
                suspicion += 0.5

        # Missing camera-specific tags is suspicious for a "photo"
        for expected in ["Make", "Model", "LensModel"]:
            if expected not in readable:
                flags.append(f"Missing EXIF field: {expected}")
                suspicion += 0.1

        gps = readable.get("GPSInfo", None)
        if not gps:
            flags.append("No GPS data (inconclusive)")

    else:
        has_exif = False
        flags.append("No EXIF data – metadata was stripped (common in deepfakes)")
        suspicion += 0.25

    # Check PIL mode inconsistency
    if pil_img.mode not in ("RGB", "L"):
        flags.append(f"Unusual image mode: {pil_img.mode}")
        suspicion += 0.1

    suspicion = min(suspicion, 1.0)
    return {
        "has_exif" : has_exif,
        "software" : software,
        "flags"    : flags,
        "suspicion": suspicion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Combined Forensic Score
# ─────────────────────────────────────────────────────────────────────────────

def run_full_forensics(pil_img: Image.Image) -> dict:
    """
    Run all three forensic analyses and return a combined dict.
    """
    ela_img,   ela_score       = compute_ela(pil_img)
    noise_img, noise_score     = compute_noise_map(pil_img)
    meta                       = inspect_metadata(pil_img)

    # Weighted ensemble forensic score (independent of CNN)
    forensic_score = (
        0.40 * ela_score +
        0.35 * noise_score +
        0.25 * meta["suspicion"]
    )
    forensic_score = min(forensic_score, 1.0)

    return {
        "ela_img"        : ela_img,
        "ela_score"      : ela_score,
        "noise_img"      : noise_img,
        "noise_score"    : noise_score,
        "metadata"       : meta,
        "forensic_score" : forensic_score,
    }
from typing import Tuple
import numpy as np
import cv2
from PIL import Image, ImageOps
import imagehash

from db import CardRow

def _clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def preprocess_for_hash(img_pil: Image.Image, max_side: int = 900) -> Image.Image:
    """Resize, trim tiny border, local-contrast normalize (CLAHE) — helps robustness."""
    w, h = img_pil.size
    scale = float(max_side) / max(w, h)
    if scale < 1.0:
        img_pil = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # Trim 3.5% border to reduce sleeve/background effects
    dx = int(img_pil.width * 0.035)
    dy = int(img_pil.height * 0.035)
    img_pil = img_pil.crop((dx, dy, img_pil.width - dx, img_pil.height - dy))

    # CLAHE on luma
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y   = _clahe_gray(yuv[:,:,0])
    yuv[:,:,0] = y
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def compute_hashes_pil(img: Image.Image) -> Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash]:
    # IMPORTANT: default hash_size=8 to match what you stored in Supabase
    return (
        imagehash.average_hash(img),
        imagehash.whash(img),
        imagehash.phash(img),
        imagehash.dhash(img),
    )

def score_max_of_mins(
    captured: Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash],
    row: CardRow
) -> int:
    cap_avg, cap_wh, cap_ph, cap_dh = captured
    d_avg = min(cap_avg - h for h in row.avg)
    d_wh  = min(cap_wh  - h for h in row.wh)
    d_ph  = min(cap_ph  - h for h in row.ph)
    d_dh  = min(cap_dh  - h for h in row.dh)
    return int(max(d_avg, d_wh, d_ph, d_dh))

def score_to_confidence(score: int, cutoff: int) -> float:
    if score is None: return 0.0
    # Map distance to [0..1]; tune denominator if desired
    return max(0.0, min(1.0, 1.0 - (score / (cutoff * 1.5))))

import os, io, ssl, certifi, urllib.request
from typing import List, Optional, Tuple
from dataclasses import dataclass

# --- TLS bootstrap for macOS/ssl issues ---
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _SSL_CTX
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))
)

# --- third party ---
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import imagehash
import math
import base64
import numpy as np

# ===================== CONFIG =====================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
TABLE         = os.getenv("SUPABASE_TABLE", "destinedrivals")
SET_FILTER    = os.getenv("SET_FILTER") or None
CUTOFF        = int(os.getenv("CUTOFF", "18"))
TOPK_DEFAULT  = int(os.getenv("TOPK", "5"))

# canonical Pokémon card aspect (~63x88 mm ⇒ ≈1.39)
CARD_W, CARD_H = 672, 936
ASPECT_MIN, ASPECT_MAX = 1.30, 1.52     # accept some tolerance around 1.39
MIN_AREA_FRAC, MAX_AREA_FRAC = 0.05, 0.95
# ==================================================

app = FastAPI(title="Razz Matcher (model-style)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# ---------- DB row in memory ----------
@dataclass
class CardRow:
    card_id: str
    name: str
    set_name: str
    ext_number: str
    subtype_name: str
    image_path: str
    avg: List[imagehash.ImageHash]
    wh:  List[imagehash.ImageHash]
    ph:  List[imagehash.ImageHash]
    dh:  List[imagehash.ImageHash]

def _hx(s: str) -> imagehash.ImageHash:
    return imagehash.hex_to_hash(str(s))

CARDS: List[CardRow] = []
SB: Optional[Client] = None


# ===== Improved normalization for hashing =====
def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _center_crop_border(img: Image.Image, border_frac: float = 0.035) -> Image.Image:
    """Trim a small uniform border (3.5% default) to remove edge/sleeve influence."""
    w, h = img.size
    dx = int(w * border_frac); dy = int(h * border_frac)
    return img.crop((dx, dy, w - dx, h - dy))

def prepare_card_for_hash(img_pil: Image.Image, max_side: int = 900) -> Image.Image:
    """
    Resize, contrast-normalize (CLAHE), and trim borders—keeps behavior stable vs glare/edges.
    """
    # 1) resize
    w, h = img_pil.size
    scale = float(max_side) / max(w, h)
    if scale < 1.0:
        img_pil = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # 2) border trim
    img_pil = _center_crop_border(img_pil, border_frac=0.035)

    # 3) light denoise + CLAHE on luma
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:,:,0]
    y = _clahe_gray(y)
    yuv[:,:,0] = y
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))




# utils_match.py


# ---- PREPROCESS (gentle denoise + contrast + standard size) ----
def preprocess_for_hash(bgr_or_rgb: Image.Image) -> Image.Image:
    img = bgr_or_rgb.convert("RGB")
    # center-crop to card-ish aspect if ROI is sloppy (optional)
    # img = ImageOps.fit(img, (512, 720), method=Image.LANCZOS)
    img = img.resize((512, 512), Image.LANCZOS)
    # mild denoise/sharpen tradeoff
    img = img.filter(ImageFilter.MedianFilter(size=3))
    # improve local contrast (CLAHE-like)
    img = ImageOps.autocontrast(img, cutoff=1)
    return img

def compute_hashes(img: Image.Image):
    # IMPORTANT: keep hash_size=8 to match what you wrote to Supabase
    def hs(i):
        return (
            imagehash.average_hash(i, hash_size=8),
            imagehash.whash(i, hash_size=8),
            imagehash.phash(i, hash_size=8),
            imagehash.dhash(i, hash_size=8),
        )
    n  = hs(img)
    m  = hs(ImageOps.mirror(img))
    ud = hs(img.transpose(Image.ROTATE_180))
    um = hs(ImageOps.mirror(img.transpose(Image.ROTATE_180)))
    return {"n": n, "m": m, "ud": ud, "um": um}

def score_record(hashes_scan, rec):
    # convert DB hex -> hashes once
    def hh(s): return imagehash.hex_to_hash(s)
    # for each method take the MIN across orientations, then MAX across methods
    # method order: avg, w, p, d
    s_avg = min(
        hashes_scan["n"][0] - hh(rec["avghashes"]),
        hashes_scan["m"][0] - hh(rec["avghashesmir"]),
        hashes_scan["ud"][0] - hh(rec["avghashesud"]),
        hashes_scan["um"][0] - hh(rec["avghashesudmir"]),
    )
    s_w = min(
        hashes_scan["n"][1] - hh(rec["whashes"]),
        hashes_scan["m"][1] - hh(rec["whashesmir"]),
        hashes_scan["ud"][1] - hh(rec["whashesud"]),
        hashes_scan["um"][1] - hh(rec["whashesudmir"]),
    )
    s_p = min(
        hashes_scan["n"][2] - hh(rec["phashes"]),
        hashes_scan["m"][2] - hh(rec["phashesmir"]),
        hashes_scan["ud"][2] - hh(rec["phashesud"]),
        hashes_scan["um"][2] - hh(rec["phashesudmir"]),
    )
    s_d = min(
        hashes_scan["n"][3] - hh(rec["dhashes"]),
        hashes_scan["m"][3] - hh(rec["dhashesmir"]),
        hashes_scan["ud"][3] - hh(rec["dhashesud"]),
        hashes_scan["um"][3] - hh(rec["dhashesudmir"]),
    )
    # fusion: max of method mins (model project style)
    score = max(s_avg, s_w, s_p, s_d)
    return int(score)

def score_to_confidence(score: Optional[int], cutoff: int) -> float:
    if score is None:
        return 0.0
    k = 0.8  # steeper makes it more “binary”
    return 1.0 / (1.0 + math.exp(k * (score - cutoff)))


def fetch_all_cards():
    """Load all rows + parse their 16 hash strings into ImageHash objects."""
    global CARDS
    CARDS = []
    page, off = 1000, 0
    while True:
        q = SB.table(TABLE).select(
            "card_id,name,set_name,ext_number,subtype_name,image_path,"
            "avghashes,avghashesmir,avghashesud,avghashesudmir,"
            "whashes,whashesmir,whashesud,whashesudmir,"
            "phashes,phashesmir,phashesud,phashesudmir,"
            "dhashes,dhashesmir,dhashesud,dhashesudmir"
        ).range(off, off+page-1)
        if SET_FILTER:
            q = q.eq("set_name", SET_FILTER)
        data = q.execute().data or []
        if not data: break
        for r in data:
            try:
                CARDS.append(CardRow(
                    card_id=r["card_id"],
                    name=r["name"],
                    set_name=r["set_name"],
                    ext_number=r["ext_number"],
                    subtype_name=r["subtype_name"],
                    image_path=r["image_path"],
                    avg=[_hx(r["avghashes"]), _hx(r["avghashesmir"]), _hx(r["avghashesud"]), _hx(r["avghashesudmir"])],
                    wh =[ _hx(r["whashes"]),  _hx(r["whashesmir"]),  _hx(r["whashesud"]),  _hx(r["whashesudmir"])],
                    ph =[ _hx(r["phashes"]),  _hx(r["phashesmir"]),  _hx(r["phashesud"]),  _hx(r["phashesudmir"])],
                    dh =[ _hx(r["dhashes"]),  _hx(r["dhashesmir"]),  _hx(r["dhashesud"]),  _hx(r["dhashesudmir"])],
                ))
            except Exception:
                # Skip malformed rows instead of crashing
                pass
        off += page
        if len(data) < page: break
    print(f"Loaded {len(CARDS)} cards from Supabase.")

# ---------- Model-like segmentation ----------
def order_corners(pts4: np.ndarray) -> np.ndarray:
    pts = np.array(pts4, dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype="float32")

def biggest_quad(contours, frame_area: int) -> Optional[np.ndarray]:
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_FRAC*frame_area or area > MAX_AREA_FRAC*frame_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:           # need 4-point contour
            continue
        if not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4,2).astype("float32")
        # aspect filter (orientation-agnostic)
        w1 = np.linalg.norm(quad[1]-quad[0]); w2 = np.linalg.norm(quad[2]-quad[3])
        h1 = np.linalg.norm(quad[3]-quad[0]); h2 = np.linalg.norm(quad[2]-quad[1])
        w = (w1 + w2)/2.0; h = (h1 + h2)/2.0
        if w == 0 or h == 0: continue
        aspect = max(h/w, w/h)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        if area > best_area:
            best_area = area
            best = quad
    return best

def segment_card_like_model(bgr_image: np.ndarray) -> Optional[np.ndarray]:
    """
    main.py-like: gray -> blur -> edges -> morph -> biggest quad -> warp (CARD_W x CARD_H).
    Now two passes: fixed Canny then adaptive threshold fallback.
    """
    H, W = bgr_image.shape[:2]

    def try_pass(edges: np.ndarray) -> Optional[np.ndarray]:
        k = np.ones((5,5), np.uint8)
        dial = cv2.dilate(edges, k, iterations=2)
        thr  = cv2.erode(dial, k, iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quad = biggest_quad(contours, H*W)
        if quad is None:
            return None
        quad = order_corners(quad)
        dst  = np.array([[0,0],[CARD_W-1,0],[CARD_W-1,CARD_H-1],[0,CARD_H-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(quad, dst)
        return cv2.warpPerspective(bgr_image, M, (CARD_W, CARD_H))

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # Pass 1: fixed Canny like model code
    edges = cv2.Canny(blur, 100, 200)
    warped = try_pass(edges)
    if warped is not None:
        return warped

    # Pass 2: adaptive threshold fallback (helps in low/high exposure)
    adap = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    edges2 = cv2.Canny(adap, 50, 150)
    return try_pass(edges2)

# ---------- Hashes + scoring (exact “max of mins”) ----------
def compute_hashes_pil(img: Image.Image, max_side: int = 900):
    """
    Normalize then compute the 4 hashes (same settings as what you stored in Supabase).
    """
    # --- normalization similar to your prepare_card_for_hash ---
    # downscale so long side == max_side (only down, never up)
    w, h = img.size
    scale = float(max_side) / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # optional small border trim to reduce sleeve/background bleed
    bw, bh = img.size
    dx = int(bw * 0.035)
    dy = int(bh * 0.035)
    img = img.crop((dx, dy, bw - dx, bh - dy))

    # light contrast normalization (helps with phone glare)
    img = ImageOps.autocontrast(img, cutoff=1).convert("RGB")

    # hashes (hash_size=8 by default — matches what you wrote to Supabase)
    return (
        imagehash.average_hash(img),
        imagehash.whash(img),
        imagehash.phash(img),
        imagehash.dhash(img),
    )



def score_max_of_mins(captured: Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash],
                      row: CardRow) -> int:
    cap_avg, cap_wh, cap_ph, cap_dh = captured
    d_avg = min(cap_avg - h for h in row.avg)
    d_wh  = min(cap_wh  - h for h in row.wh)
    d_ph  = min(cap_ph  - h for h in row.ph)
    d_dh  = min(cap_dh  - h for h in row.dh)
    return max(d_avg, d_wh, d_ph, d_dh)

# ---------- API ----------
@app.on_event("startup")
def _startup():
    global SB
    SB = create_client(SUPABASE_URL, SUPABASE_KEY)
    fetch_all_cards()

@app.get("/health")
def health():
    return {"ok": True, "cards": len(CARDS), "cutoff": CUTOFF, "set_filter": SET_FILTER}

@app.post("/match")
async def match(
    file: UploadFile = File(...),
    strategy: str = Form("roi"),
    max_side: int = Form(900),
    top_k: int = Form(TOPK_DEFAULT),
    cutoff: int = Form(CUTOFF),
):
    raw = await file.read()

    # Load with Pillow, honor EXIF rotation
    img_pil = Image.open(io.BytesIO(raw))
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")

    if strategy == "auto":
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        warped = segment_card_like_model(bgr)
        if warped is None:
            return {
                "match": None, "best": None, "top": [],
                "cutoff": cutoff, "count": len(CARDS), "error": "no_card_detected"
            }
        img_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    # Hash once with internal normalization
    cap_hashes = compute_hashes_pil(img_pil, max_side=max_side)

    # Score all rows with exact “max of mins”
    scored = []
    for r in CARDS:
        s = score_max_of_mins(cap_hashes, r)
        scored.append((s, r))
    scored.sort(key=lambda x: x[0])

    top = [{
        "score": s,
        "card_id": r.card_id,
        "name": r.name,
        "set_name": r.set_name,
        "ext_number": r.ext_number,
        "subtype_name": r.subtype_name,
        "image_path": r.image_path,
    } for s, r in scored[:max(1, top_k)]]

    best = top[0] if top else None
    best_score = best["score"] if best else None
    is_confident = bool(best and best_score is not None and best_score < cutoff)
    confidence = score_to_confidence(best_score, cutoff)

    return {
        "match": (best if is_confident else None),
        "best": best,
        "top": top,
        "best_score": best_score,
        "is_confident": is_confident,
        "confidence": confidence,
        "cutoff": cutoff,
        "count": len(CARDS),
        "mode": strategy,
    }



@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    return_warp: bool = Form(False),
    preview_side: int = Form(320),
):
    raw = await file.read()
    img_pil = Image.open(io.BytesIO(raw))
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    H, W = bgr.shape[:2]
    out = {"corners": None, "width": W, "height": H}

    # run detector but keep corners
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 100, 200)
    k = np.ones((5,5), np.uint8)
    dial = cv2.dilate(edges, k, iterations=2)
    thr  = cv2.erode(dial, k, iterations=1)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = biggest_quad(contours, H*W)
    if quad is None:
        # fallback adaptive
        adap = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        edges2 = cv2.Canny(adap, 50, 150)
        dial2 = cv2.dilate(edges2, k, iterations=2)
        thr2  = cv2.erode(dial2, k, iterations=1)
        contours, _ = cv2.findContours(thr2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quad = biggest_quad(contours, H*W)

    if quad is not None:
        quad = order_corners(quad).tolist()
        out["corners"] = quad

        if return_warp:
            dst  = np.array([[0,0],[CARD_W-1,0],[CARD_W-1,CARD_H-1],[0,CARD_H-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(np.array(quad, dtype="float32"), dst)
            warped = cv2.warpPerspective(bgr, M, (CARD_W, CARD_H))
            # small preview
            s = preview_side
            r = cv2.resize(warped, (s, int(s*CARD_H/CARD_W)))
            ok, buf = cv2.imencode(".jpg", r, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            out["warp_jpg_b64"] = base64.b64encode(buf).decode("ascii")

    return out

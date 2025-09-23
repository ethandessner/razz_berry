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
from PIL import Image, ImageOps
import imagehash

# ===================== CONFIG =====================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
TABLE         = os.getenv("SUPABASE_TABLE", "destined")
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
    Gray -> GaussianBlur -> Canny -> dilate/erode -> biggest 4-pt contour -> warp to (CARD_W x CARD_H).
    Mirrors your main.py + utils flow.
    """
    H, W = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 100, 200)
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
    warped = cv2.warpPerspective(bgr_image, M, (CARD_W, CARD_H))
    return warped

# ---------- Hashes + scoring (exact “max of mins”) ----------
def compute_hashes_pil(img: Image.Image) -> Tuple[imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash, imagehash.ImageHash]:
    img = img.convert("RGB")
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
    strategy: str = Form("roi"),            # "roi" (already cropped) or "auto" (server segments like model)
    max_side: int = Form(900),              # normalize long side before hashing
    top_k: int = Form(TOPK_DEFAULT),
    cutoff: int = Form(CUTOFF),
):
    raw = await file.read()

    # Load with Pillow, honor EXIF rotation
    img_pil = Image.open(io.BytesIO(raw))
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")

    if strategy == "auto":
        # Convert to BGR and run model-style segmentation
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        warped = segment_card_like_model(bgr)
        if warped is None:
            return {"match": None, "best": None, "top": [], "cutoff": cutoff, "count": len(CARDS), "error": "no_card_detected"}
        img_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    # Normalize image size for stable hashing
    w, h = img_pil.size
    scale = float(max_side) / max(w, h)
    if scale < 1.0:
        img_pil = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    cap_hashes = compute_hashes_pil(img_pil)

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
    return {
        "match": (best if best and best["score"] < cutoff else None),
        "best": best,
        "top": top,
        "cutoff": cutoff,
        "count": len(CARDS),
        "mode": strategy,
    }

import os, io, ssl, base64, json, certifi, urllib.request, math
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _SSL_CTX
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))
)

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from supabase import create_client, Client

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import imagehash

# CONFIG 
load_dotenv()
SUPABASE_URL   = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY   = os.getenv("SUPABASE_KEY", "").strip()
TABLE          = os.getenv("SUPABASE_TABLE", "destinedrivals")
SET_FILTER     = os.getenv("SET_FILTER") or None
CUTOFF         = int(os.getenv("CUTOFF", "18"))
TOPK_DEFAULT   = int(os.getenv("TOPK", "5"))

# Embedding configuration (Vertex multimodal embedding model)
EMBEDDINGS_COLUMN = os.getenv("EMBEDDINGS_COLUMN", "embedding_vec")
VERTEX_EMBED_MODEL = os.getenv("VERTEX_EMBED_MODEL", "multimodalembedding@001").strip()
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "cardconnect-ethandessner").strip()
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1").strip()
VERTEX_EMBED_DIM = int(os.getenv("VERTEX_EMBED_DIM", "1408"))
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "1").strip() not in ("0", "false", "False")
_vertex_model = None  # lazy-loaded model handle
EMBED_SIM_THRESHOLD = float(os.getenv("EMBED_SIM_THRESHOLD", "0.80"))  # similarity >= threshold => confident
EMBED_DISPLAY_DIMS = int(os.getenv("EMBED_DISPLAY_DIMS", "16"))  # number of dims to send/display for query embedding
PURE_EMBED_MODE = os.getenv("PURE_EMBED_MODE", "1").strip() in ("1", "true", "True")  # if true, ignore all hash columns
EMBED_USE_SEGMENTATION = os.getenv("EMBED_USE_SEGMENTATION", "1").strip() not in ("0","false","False")
EMBED_TRY_FULL_FRAME = os.getenv("EMBED_TRY_FULL_FRAME", "1").strip() not in ("0","false","False")  # if true, also compute full-frame embedding and choose better
_PCA_READY = False
_EMB_MEAN = None
_PCA_W = None  #  (d x 2) matrix for projection

# Pokémon card size/aspect (≈63x88mm)
CARD_W, CARD_H = 672, 936
ASPECT_MIN, ASPECT_MAX = 1.30, 1.52
MIN_AREA_FRAC, MAX_AREA_FRAC = 0.05, 0.95

app = FastAPI(title="Razz Matcher (auto-segment + monitor)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# DB row in memory 
@dataclass
class CardRow:
    card_id: str
    name: str
    set_name: str
    ext_number: str
    subtype_name: str
    image_path: str
    avg: List[imagehash.ImageHash] = field(default_factory=list)
    wh:  List[imagehash.ImageHash] = field(default_factory=list)
    ph:  List[imagehash.ImageHash] = field(default_factory=list)
    dh:  List[imagehash.ImageHash] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # normalized embedding vector

def _hx(s: str) -> imagehash.ImageHash:
    # Accept raw hex or ImageHash("...") forms
    return imagehash.hex_to_hash(str(s))

CARDS: List[CardRow] = []
SB: Optional[Client] = None

# WebSocket monitor pool
ACTIVE_SOCKETS: Set[WebSocket] = set()

async def broadcast_debug(payload: dict):
    dead = []
    txt = json.dumps(payload)
    for ws in list(ACTIVE_SOCKETS):
        try:
            await ws.send_text(txt)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ACTIVE_SOCKETS.discard(ws)

# Vision helpers 
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
        
        w1 = np.linalg.norm(quad[1]-quad[0]); w2 = np.linalg.norm(quad[2]-quad[3])
        h1 = np.linalg.norm(quad[3]-quad[0]); h2 = np.linalg.norm(quad[2]-quad[1])
        w = (w1 + w2)/2.0; h = (h1 + h2)/2.0
        if w == 0 or h == 0: 
            continue
        aspect = max(h/w, w/h)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        if area > best_area:
            best_area = area
            best = quad
    return best

# Segmentation tuned for clean edges and stable warping 
CARD_W, CARD_H = 330, 440  # Match utils.py
_K5 = np.ones((5,5), np.uint8)

# Returns the corners and area of the biggest contour
def biggestContour(contours):
    biggest = np.array([])
    maxArea = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest, maxArea

# Returns corners in order [topleft, topright, bottomleft, bottomright]
def reorderCorners(corners):
    xvals = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]]
    yvals = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]]
    def sortVals(vals):
        indexes = list(range(len(vals)))
        for i in range(len(vals)):
            index = i
            minval = vals[i]
            for j in range(i, len(vals)):
                if vals[j] < minval:
                    minval = vals[j]
                    index = j
            vals[i], vals[index] = vals[index], vals[i]
            indexes[i], indexes[index] = indexes[index], indexes[i]
        return vals, indexes
    yvals, idxs = sortVals(yvals)
    temp = xvals.copy()
    for i in range(len(idxs)):
        xvals[i] = temp[idxs[i]]
    # Check if card is horizontal or vertical
    if yvals[0] == yvals[1]:
        if xvals[1] < xvals[0]:
            xvals[0], xvals[1] = xvals[1], xvals[0]
    dist1 = ((xvals[1] - xvals[0]) ** 2 + (yvals[1] - yvals[0]) ** 2) ** 0.5
    dist2 = ((xvals[2] - xvals[0]) ** 2 + (yvals[2] - yvals[0]) ** 2) ** 0.5
    dist3 = ((xvals[3] - xvals[0]) ** 2 + (yvals[3] - yvals[0]) ** 2) ** 0.5
    dists = [dist1, dist2, dist3]
    distSorted, idxsDist = sortVals(dists.copy())
    idxsDist.insert(0, 0)
    idxsDist[1] += 1
    idxsDist[2] += 1
    idxsDist[3] += 1
    if yvals[0] == yvals[1]:
        if dists[0] == distSorted[0]:
            topleft = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            topright = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            bottomright = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
            bottomleft = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
        else:
            topleft = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            topright = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            bottomright = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
            bottomleft = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
    else:
        if xvals[idxsDist[1]] == min(xvals):
            topleft = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            topright = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            bottomright = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
            bottomleft = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
        else:
            topleft = [xvals[idxsDist[0]], yvals[idxsDist[0]]]
            topright = [xvals[idxsDist[1]], yvals[idxsDist[1]]]
            bottomright = [xvals[idxsDist[3]], yvals[idxsDist[3]]]
            bottomleft = [xvals[idxsDist[2]], yvals[idxsDist[2]]]
    return np.array([topleft, topright, bottomleft, bottomright], dtype="float32")

# Segmentation and transformation logic
def segment_card_like_model(bgr_image: np.ndarray):
    H, W = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 100, 200)
    kernel = np.ones((5,5), np.uint8)
    frameDial = cv2.dilate(edged, kernel, iterations=2)
    frameThreshold = cv2.erode(frameDial, kernel, iterations=1)
    contours, _ = cv2.findContours(frameThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = bgr_image.copy()
    biggest, maxArea = biggestContour(contours)
    quad = None
    if len(biggest) == 4:
        corners = [biggest[0][0], biggest[1][0], biggest[2][0], biggest[3][0]]
        quad = reorderCorners(corners)
        cv2.polylines(overlay, [quad.astype(int)], True, (0,255,0), 3)
        pts1 = np.float32(quad)
        pts2 = np.float32([[0, 0], [CARD_W, 0], [0, CARD_H], [CARD_W, CARD_H]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(bgr_image, matrix, (CARD_W, CARD_H))
    else:
        warped = np.zeros((CARD_H, CARD_W, 3), np.uint8)
    return warped, overlay, edged, frameThreshold, quad

# Hashing prep (stabilize against lighting/sleeves) - this lowk doesn't work rn (kinda)
def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def prepare_card_for_hash(img_pil: Image.Image, max_side: int = 900) -> Image.Image:
    """Same normalization for DB and live: trim border, CLAHE on luma, mild denoise, area resample."""
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # trim 3.5% uniform border (sleeves/edge lighting) - remove/fix later
    h, w = bgr.shape[:2]
    dx = int(w * 0.035); dy = int(h * 0.035)
    bgr = bgr[dy:h-dy, dx:w-dx].copy()

    # CLAHE on luminance only
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(y)
    bgr = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # mild edge-preserving denoise
    bgr = cv2.bilateralFilter(bgr, 5, 60, 60)

    # downscale (if needed) with INTER_AREA (best for shrink)
    h, w = bgr.shape[:2]
    scale = float(max_side) / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def compute_hashes_pil(img: Image.Image):
    img = prepare_card_for_hash(img, max_side=900).convert("RGB")
    return (
        imagehash.average_hash(img),  # size=8 default - need to increase this
        imagehash.whash(img),
        imagehash.phash(img),
        imagehash.dhash(img),
    )

def score_max_of_mins(captured, row: CardRow) -> int:
    cap_avg, cap_wh, cap_ph, cap_dh = captured
    d_avg = min(cap_avg - h for h in row.avg)
    d_wh  = min(cap_wh  - h for h in row.wh)
    d_ph  = min(cap_ph  - h for h in row.ph)
    d_dh  = min(cap_dh  - h for h in row.dh)
    return int(max(d_avg, d_wh, d_ph, d_dh))

def score_to_confidence(score: Optional[int], cutoff: int) -> float:
    if score is None: return 0.0
    # 1.0 at 0 distance, fades toward 0 near ~1.5*cutoff
    return max(0.0, min(1.0, 1.0 - (score / (cutoff * 1.5))))


def fetch_all_cards():
    """Load rows + parse hashes + optional embedding vectors from Supabase."""
    global CARDS
    CARDS = []
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Supabase env missing; skipping preload.")
        return

    embed_col = EMBEDDINGS_COLUMN
    page, off = 1000, 0
    if PURE_EMBED_MODE:
        base_select = "card_id,name,set_name,ext_number,subtype_name,image_path"
    else:
        base_select = (
            "card_id,name,set_name,ext_number,subtype_name,image_path," \
            "avghashes,avghashesmir,avghashesud,avghashesudmir," \
            "whashes,whashesmir,whashesud,whashesudmir," \
            "phashes,phashesmir,phashesud,phashesudmir," \
            "dhashes,dhashesmir,dhashesud,dhashesudmir"
        )
    select_clause = base_select + (f",{embed_col}" if USE_EMBEDDINGS else "")
    while True:
        q = SB.table(TABLE).select(select_clause).range(off, off+page-1)
        if SET_FILTER:
            q = q.eq("set_name", SET_FILTER)
        resp = q.execute()
        data = resp.data or []
        if not data:
            break
        for r in data:
            try:
                emb_vec = None
                if USE_EMBEDDINGS and embed_col in r and r[embed_col] is not None:
                    rawv = r[embed_col]
                    if isinstance(rawv, str):
                        try:
                            rawv = json.loads(rawv)
                        except Exception:
                            rawv = None
                    if isinstance(rawv, (list, tuple)):
                        arr = np.array(rawv, dtype=float)
                        n = np.linalg.norm(arr) or 1.0
                        emb_vec = (arr / n).astype(float).tolist()
                if PURE_EMBED_MODE:
                    CARDS.append(CardRow(
                        card_id=r["card_id"], name=r["name"], set_name=r["set_name"],
                        ext_number=r["ext_number"], subtype_name=r["subtype_name"], image_path=r["image_path"],
                        embedding=emb_vec
                    ))
                else:
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
                        embedding=emb_vec,
                    ))
            except Exception as e:
                print("row skip:", e)
        off += page
        if len(data) < page:
            break
    have = sum(1 for c in CARDS if c.embedding)
    print(f"Loaded {len(CARDS)} cards from Supabase ({TABLE}); {have} with embeddings.")
    if have >= 5:
        _build_pca_projection()
    else:
        print("[pca] Not enough embeddings for PCA visualization (need >=5)")

def _build_pca_projection():
    """Compute a quick 2D PCA projection of current embeddings for monitor plotting."""
    global _PCA_READY, _EMB_MEAN, _PCA_W
    emb_list = [c.embedding for c in CARDS if c.embedding]
    if len(emb_list) < 5:
        _PCA_READY = False
        return
    X = np.array(emb_list, dtype=float)
    _EMB_MEAN = X.mean(axis=0)
    Xc = X - _EMB_MEAN
    # covariance via SVD for speed with possibly high dim vectors
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        _PCA_W = Vt[:2].T  # d x 2
        _PCA_READY = True
        print(f"[pca] Projection ready (embeddings={len(emb_list)})")
    except Exception as e:
        print(f"[pca] SVD failed: {e}")
        _PCA_READY = False

def _pca_project(vec: List[float]):
    if not _PCA_READY or _PCA_W is None or _EMB_MEAN is None:
        return None
    v = np.array(vec, dtype=float) - _EMB_MEAN
    pt = v @ _PCA_W  # 2D
    return pt.tolist()

# ---------------- Embedding Runtime (Vertex) -----------------
def _init_vertex_model():
    global _vertex_model
    if _vertex_model is not None:
        return
    if not USE_EMBEDDINGS:
        _vertex_model = False
        return
    if not VERTEX_PROJECT_ID:
        print("[embed] VERTEX_PROJECT_ID missing; disabling embeddings")
        _vertex_model = False
        return
    try:
        import vertexai  # type: ignore
        from vertexai.vision_models import MultiModalEmbeddingModel  # type: ignore
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        _vertex_model = MultiModalEmbeddingModel.from_pretrained(VERTEX_EMBED_MODEL)
        print(f"[embed] Loaded model {VERTEX_EMBED_MODEL} @ {VERTEX_PROJECT_ID}/{VERTEX_LOCATION}")
    except Exception as e:
        print(f"[embed] init failed: {e}")
        _vertex_model = False

def _l2(v: List[float]):
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def _embed_pil(img: Image.Image) -> Optional[List[float]]:
    if not USE_EMBEDDINGS:
        return None
    if _vertex_model is None:
        _init_vertex_model()
    if not _vertex_model or _vertex_model is False:
        return None
    try:
        from vertexai.vision_models import Image as VImage  # type: ignore
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        vimg = VImage(image_bytes=buf.getvalue())
        params = {"image": vimg}
        if VERTEX_EMBED_DIM:
            params["dimension"] = VERTEX_EMBED_DIM
        out = _vertex_model.get_embeddings(**params)
        vec = list(out.image_embedding)
        if not vec:
            return None
        return _l2(vec)
    except Exception as e:
        print(f"[embed] fail: {e}")
        return None

def _nearest_by_embedding(query_vec: List[float], top_k: int) -> List[Tuple[float, CardRow]]:
    q = np.array(query_vec, dtype=float)
    # Cosine distance since vectors are normalized: dist = 1 - dot
    scored: List[Tuple[float, CardRow]] = []
    for c in CARDS:
        if not c.embedding:
            continue
        v = np.array(c.embedding, dtype=float)
        sim = float(np.dot(q, v))  # cosine similarity
        # higher is better; we'll sort descending
        scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:max(1, top_k)]

def _pca_all_points():
    if not _PCA_READY:
        return []
    pts = []
    for c in CARDS:
        if c.embedding:
            proj = _pca_project(c.embedding)
            if proj:
                pts.append(proj)
    return pts

# ---------- API ----------
@app.on_event("startup")
def _startup():
    global SB
    if SUPABASE_URL and SUPABASE_KEY:
        SB = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        SB = None
    fetch_all_cards()

@app.get("/health")
def health():
    return {"ok": True, "cards": len(CARDS), "cutoff": CUTOFF, "set_filter": SET_FILTER}

@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    # Simple page that connects to /ws and shows the latest montage + info
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Razz Monitor</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; background:#0b0b0b; color:#eee; }
  #wrap { display:flex; gap:16px; align-items:flex-start; }
  #grid { max-width: 70vw; border-radius: 10px; overflow:hidden; border:1px solid #333; }
  #meta { min-width: 300px; }
  .tag { display:inline-block; padding:2px 8px; border-radius:999px; background:#1f1f1f; margin-right:6px; font-size:12px; }
  img { display:block; width:100%; height:auto; }
  code { background:#1f1f1f; padding:2px 6px; border-radius:6px; }
</style>
</head>
<body>
  <h2>Razz Live Monitor</h2>
  <div id="wrap">
    <div id="grid"><img id="dbg" alt="debug montage"/></div>
    <div id="meta">
      <div id="status" class="tag">waiting…</div>
      <div style="height:8px"></div>
      <div>Best: <span id="best">–</span></div>
      <div>Score: <code id="score">–</code></div>
      <div>Confidence: <code id="conf">–</code></div>
            <div>Cutoff: <code id="cutoff">–</code>  |  Sim thresh: <code id="simthresh">–</code></div>
      <div>Corners: <code id="corners">–</code></div>
    <div style="height:8px"></div>
    <div>Embedding dims: <code id="embdims">–</code></div>
    <div style="font-size:12px; line-height:1.3; max-width:320px; word-break:break-all;">q[0..]: <code id="embpreview">–</code></div>
      <div style="height:12px"></div>
      <div>Top:</div>
      <ol id="top"></ol>
            <div style="height:16px"></div>
            <div>Embedding space (PCA 2D):</div>
            <canvas id="pcacanvas" width="300" height="300" style="border:1px solid #333; background:#111; border-radius:6px"></canvas>
            <div id="simlist" style="margin-top:8px; font-size:12px; line-height:1.3"></div>
    </div>
  </div>
<script>
  function wsURL(){
    const l = window.location;
    const scheme = (l.protocol === 'https:') ? 'wss' : 'ws';
    return scheme + '://' + l.host + '/ws';
  }
  const ws = new WebSocket(wsURL());
  const $img = document.getElementById('dbg');
  const $status = document.getElementById('status');
  const $best = document.getElementById('best');
  const $score = document.getElementById('score');
  const $conf = document.getElementById('conf');
  const $cutoff = document.getElementById('cutoff');
  const $corners = document.getElementById('corners');
    const $embdims = document.getElementById('embdims');
    const $embpreview = document.getElementById('embpreview');
    const $top = document.getElementById('top');
    const $simthresh = document.getElementById('simthresh');
    const $canvas = document.getElementById('pcacanvas');
    const ctx = $canvas.getContext('2d');
    const $simlist = document.getElementById('simlist');

    function drawPCA(payload){
        if(!payload || !payload.pca){ ctx.clearRect(0,0,$canvas.width,$canvas.height); return; }
        const {points, query, neighbors} = payload.pca;
        ctx.clearRect(0,0,$canvas.width,$canvas.height);
        if(!points || points.length === 0) return;
        // compute bounds
        let xs = [], ys = [];
        points.forEach(p=>{ xs.push(p[0]); ys.push(p[1]); });
        if(query) { xs.push(query[0]); ys.push(query[1]); }
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);
        const pad = 0.08; // 8% padding
        const dx = (maxX - minX) || 1; const dy = (maxY - minY) || 1;
        function toPix(pt){
            return [
                ( (pt[0]-minX)/dx * (1-pad*2) + pad) * $canvas.width,
                (1 - (pt[1]-minY)/dy * (1-pad*2) - pad) * $canvas.height
            ];
        }
        // plot background points (all embeddings)
        ctx.fillStyle = '#444';
        points.forEach(pt=>{
            const [x,y] = toPix(pt);
            ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
        });
        // neighbors
        if(neighbors){
            ctx.fillStyle = '#1e88e5';
            neighbors.forEach(pt=>{ const [x,y]=toPix(pt); ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fill(); });
        }
        // query point
        if(query){
            const [xq,yq] = toPix(query);
            ctx.fillStyle = '#ffcc00';
            ctx.beginPath(); ctx.arc(xq,yq,6,0,Math.PI*2); ctx.fill();
            ctx.strokeStyle = '#fff'; ctx.lineWidth=2; ctx.stroke();
        }
    }

    function renderSims(top){
        if(!top){ $simlist.innerHTML=''; return; }
        $simlist.innerHTML = top.map(t=>{
            const pct = (t.similarity*100).toFixed(1);
            return `<div style="display:flex;align-items:center;gap:6px;">\n`+
                         `<div style="flex:0 0 42px; color:#aaa;">${pct}%</div>`+
                         `<div style="flex:1; background:#222; height:6px; border-radius:4px; overflow:hidden;">`+
                         `<div style="width:${pct}%; height:100%; background:#4caf50"></div></div>`+
                         `<div style="flex:0 0 auto; color:#888; font-size:10px;">${t.card_id}</div>`+
                         `</div>`;
        }).join('');
    }

  ws.onopen = () => { $status.textContent = 'connected'; };
  ws.onclose = () => { $status.textContent = 'disconnected'; };
  ws.onerror = () => { $status.textContent = 'error'; };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.jpg_b64) {
      $img.src = 'data:image/jpeg;base64,' + msg.jpg_b64;
    }
    $cutoff.textContent = msg.cutoff ?? '-';
    $corners.textContent = JSON.stringify(msg.corners ?? null);
    $simthresh.textContent = msg.similarity_threshold ?? '-';
        if (msg.query_embedding) {
            $embdims.textContent = msg.query_embedding.dim || '-';
            $embpreview.textContent = (msg.query_embedding.first || []).map(v=>v.toFixed(4)).join(', ');
        } else {
            $embdims.textContent = '–';
            $embpreview.textContent = '–';
        }
        if (msg.best) {
            // Confident match
            $best.textContent = `${msg.best.name} • ${msg.best.set_name} • ${msg.best.ext_number} • ${msg.best.subtype_name}`;
            $score.textContent = String(msg.best.score);
            $conf.textContent = ((msg.confidence ?? 0)*100).toFixed(0) + '%';
        } else if (msg.raw_best) {
            const simPct = ((msg.raw_best.similarity||0)*100).toFixed(1);
            $best.textContent = `No confident match (top ${simPct}% sim)`;
            $score.textContent = String(msg.raw_best.score ?? '–');
            $conf.textContent = simPct + '%';
        } else {
            $best.textContent = '–';
            $score.textContent = '–';
            $conf.textContent = '–';
        }
    $top.innerHTML = '';
        renderSims(msg.top || []);
        (msg.top || []).forEach((t) => {
      const li = document.createElement('li');
      li.textContent = `${t.score}  ${t.card_id}`;
      $top.appendChild(li);
    });
        drawPCA(msg);
  };
</script>
</body>
</html>
    """.strip()

@app.websocket("/ws")
async def ws_debug(ws: WebSocket):
    await ws.accept()
    ACTIVE_SOCKETS.add(ws)
    try:
        while True:
            # We don’t expect messages from the client; just keep connection alive.
            await ws.receive_text()
    except WebSocketDisconnect:
        ACTIVE_SOCKETS.discard(ws)
    except Exception:
        ACTIVE_SOCKETS.discard(ws)



@app.post("/match")
async def match(
    file: UploadFile = File(...),
    strategy: str = Form("auto"),         # always auto-segment
    top_k: int = Form(TOPK_DEFAULT),
    cutoff: int = Form(CUTOFF),            # kept for backward compat; not used in embedding mode
):
    try:
        raw = await file.read()
        img_pil = Image.open(io.BytesIO(raw))
        img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")

        # Original BGR frame
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Optional segmentation
        warped = None; overlay = bgr.copy(); edges = np.zeros_like(bgr); thr = np.zeros_like(bgr); quad = None
        if EMBED_USE_SEGMENTATION:
            warped, overlay, edges, thr, quad = segment_card_like_model(bgr)
        # If segmentation disabled or failed (warped is blank or mostly zeros), we'll rely on full-frame path.
        use_seg = False
        if EMBED_USE_SEGMENTATION and warped is not None:
            # Heuristic: check if warped has variance (not pure blank from fail case)
            if np.var(warped) > 5.0:  # arbitrary small threshold
                use_seg = True

        # Prepare candidate images for embedding
        seg_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)) if use_seg else None
        full_pil = img_pil  # original (already RGB)
        # Optionally resize full frame to card aspect & size for consistency
        if EMBED_TRY_FULL_FRAME:
            full_pil_resized = full_pil.resize((CARD_W, CARD_H), Image.BILINEAR)
        else:
            full_pil_resized = None

        # Embedding-only path
        best = None
        top = []
        best_score = None  # pseudo "distance" = 1 - similarity
        confidence = 0.0
        is_confident = False

        chosen_mode = None
        qvec = None
        scored = []
        alt_info = {}
        if USE_EMBEDDINGS:
            seg_vec = _embed_pil(seg_pil) if seg_pil else None
            full_vec = _embed_pil(full_pil_resized) if full_pil_resized else None
            # Decide which vector to prefer:
            # Strategy: if both exist run quick top-1 similarity for each and choose higher; else fallback to whichever exists.
            best_seg = None; best_full = None
            if seg_vec:
                seg_scored_tmp = _nearest_by_embedding(seg_vec, 1)
                if seg_scored_tmp:
                    best_seg = float(seg_scored_tmp[0][0])
            if full_vec:
                full_scored_tmp = _nearest_by_embedding(full_vec, 1)
                if full_scored_tmp:
                    best_full = float(full_scored_tmp[0][0])
            # Selection logic
            if seg_vec and full_vec:
                if best_full is not None and best_seg is not None and best_full > best_seg + 0.005:  # small margin
                    qvec = full_vec; chosen_mode = "full"
                else:
                    qvec = seg_vec; chosen_mode = "seg"
            elif seg_vec:
                qvec = seg_vec; chosen_mode = "seg"
            elif full_vec:
                qvec = full_vec; chosen_mode = "full"
            else:
                qvec = None
            if qvec:
                scored = _nearest_by_embedding(qvec, top_k)
            alt_info = {
                "variant_scores": {
                    "seg_top1": best_seg,
                    "full_top1": best_full
                },
                "variant_used": chosen_mode,
                "segmentation_used": bool(use_seg)
            }
        if qvec:
            top = [{
                "similarity": float(sim),
                "score": float(1.0 - sim),
                "card_id": r.card_id,
                "name": r.name,
                "set_name": r.set_name,
                "ext_number": r.ext_number,
                "subtype_name": r.subtype_name,
                "image_path": r.image_path,
            } for sim, r in scored]
            top = [{
                **t,
                "mode": chosen_mode
            } for t in top]
            best = top[0] if top else None
            if best:
                best_score = best["score"]
                confidence = max(0.0, min(1.0, best["similarity"]))
                is_confident = best["similarity"] >= EMBED_SIM_THRESHOLD
        else:
            # No embedding produced; return empty result (no hash fallback by design)
            return {
                "match": None,
                "best": None,
                "top": [],
                "best_score": None,
                "is_confident": False,
                "confidence": 0.0,
                "cutoff": cutoff,
                "count": len(CARDS),
                "mode": "embedding",
                "error": "embedding_unavailable"
            }

        # Push live montage to /monitor
        montage_jpg = _make_montage_jpg(bgr, overlay, edges, thr, warped, best)
        debug_payload = {
            "jpg_b64": montage_jpg,
            "corners": quad.tolist() if quad is not None else None,
            # Only surface a 'best' if confident; always send raw_best for debugging
            "best": (best if is_confident else None),
            "raw_best": best,
            "top": top[:5],
            "cutoff": cutoff,
            "confidence": confidence,
            "similarity_threshold": EMBED_SIM_THRESHOLD,
            "is_confident": is_confident,
            **({"query_embedding": {
                "dim": len(qvec),
                "first": qvec[:EMBED_DISPLAY_DIMS]
            }} if qvec else {}),
            **({"pca": {
                "points": _pca_all_points(),
                "query": (_pca_project(qvec) if qvec else None),
                "neighbors": [ _pca_project(c.embedding) for c in [s[1] for s in scored[:5]] if _PCA_READY and c.embedding ]
            }} if qvec and _PCA_READY else {})
        }
        debug_payload.update(alt_info)
        await broadcast_debug(debug_payload)

        # Response for the app
        response_payload = {
            "match": (best if is_confident else None),
            "best": best,
            "top": top,
            "best_score": best_score,           # distance-style (lower better)
            "best_similarity": (best.get("similarity") if best else None),
            "is_confident": is_confident,
            "confidence": confidence,           # same as best_similarity capped 0..1
            "similarity_threshold": EMBED_SIM_THRESHOLD,
            "cutoff": cutoff,                   # legacy field (unused in embedding)
            "count": len(CARDS),
            "mode": "embedding",
            **({"query_embedding": {
                "dim": len(qvec),
                "first": qvec[:EMBED_DISPLAY_DIMS]
            }} if qvec else {}),
            **({"pca": {
                "points": _pca_all_points(),
                "query": (_pca_project(qvec) if qvec else None),
                "neighbors": [ _pca_project(c.embedding) for c in [s[1] for s in scored[:top_k]] if _PCA_READY and c.embedding ]
            }} if qvec and _PCA_READY else {})
        }
        response_payload.update(alt_info)
        return response_payload

    except Exception as e:
        # Return 200 with an error payload so the app can show a toast instead of just failing
        return JSONResponse(status_code=200, content={
            "match": None, "best": None, "top": [],
            "cutoff": cutoff, "count": len(CARDS),
            "error": f"internal_error: {type(e).__name__}: {str(e)}"
        })


def _encode_jpg(img_bgr, q=80) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return base64.b64encode(buf).decode("ascii") if ok else ""

def _tile(*imgs):
    # Resize to same height and hstack
    hs = 320
    outs = []
    for im in imgs:
        if im is None:
            outs.append(np.zeros((hs, int(hs*0.75), 3), np.uint8))
            continue
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        r = hs / float(h)
        outs.append(cv2.resize(im, (int(w*r), hs)))
    return cv2.hconcat(outs)

def _make_montage_jpg(orig_bgr, overlay_bgr, edges, thr, warped_bgr, best):
    row1 = _tile(orig_bgr, overlay_bgr, edges)
    row2 = _tile(thr, warped_bgr)
    # pad second row to same width as first
    w1 = row1.shape[1]; w2 = row2.shape[1]
    if w2 < w1:
        pad = np.zeros((row2.shape[0], w1 - w2, 3), np.uint8)
        row2 = cv2.hconcat([row2, pad])
    grid = cv2.vconcat([row1, row2])

    text = "Razz Berry Monitor"
    if best:
        text += f"  |  Best: {best.get('name','')}  score {best.get('score')}"
    cv2.rectangle(grid, (0,0), (grid.shape[1], 30), (32,32,32), -1)
    cv2.putText(grid, text, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    return _encode_jpg(grid, q=80)

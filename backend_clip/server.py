from __future__ import annotations
import io, base64, time
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import numpy as np

from config import TOPK, EMBED_COL, SUPABASE_TABLE
from supabase_utils import fetch_cards, CardRow
from vertex_embed import embed_image

app = FastAPI(title="Embedding Matcher (Vertex AI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

CARD_CACHE: List[CardRow] = []
EMB_MATRIX: Optional[np.ndarray] = None  # shape (N, D)
CARD_ID_INDEX: List[str] = []


class MatchResult(BaseModel):
    card_id: str
    name: str
    set_name: str
    ext_number: str
    subtype_name: str
    score: float  # cosine similarity

class MatchResponse(BaseModel):
    top: List[MatchResult]
    best: Optional[MatchResult]
    count: int
    dim: Optional[int]
    elapsed_ms: float


def _load_cache():
    global CARD_CACHE, EMB_MATRIX, CARD_ID_INDEX
    CARD_CACHE = fetch_cards(missing_only=False)
    # Filter only cards with embeddings
    avail = [c for c in CARD_CACHE if c.embedding]
    if not avail:
        EMB_MATRIX = None
        CARD_ID_INDEX = []
        print("[cache] No embeddings loaded.")
        return
    mat = np.array([c.embedding for c in avail], dtype=np.float32)
    # Ensure already normalized (should be) – but re-normalize defensively
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    EMB_MATRIX = mat
    CARD_ID_INDEX = [c.card_id for c in avail]
    print(f"[cache] Loaded {len(avail)} embeddings with dim={mat.shape[1]}")


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_cache()
    yield

app = FastAPI(title="Embedding Matcher (Vertex AI)", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "cards_total": len(CARD_CACHE),
        "with_embeddings": 0 if EMB_MATRIX is None else EMB_MATRIX.shape[0],
        "dim": None if EMB_MATRIX is None else EMB_MATRIX.shape[1],
        "table": SUPABASE_TABLE,
        "embed_col": EMBED_COL,
    }


@app.post("/refresh")
def refresh():
    _load_cache()
    return {"ok": True, "with_embeddings": 0 if EMB_MATRIX is None else EMB_MATRIX.shape[0]}


@app.post("/match", response_model=MatchResponse)
async def match(
    file: UploadFile = File(...),
    top_k: int = Form(TOPK),
):
    t0 = time.time()
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")

    q_vec = embed_image(img)  # normalized vector
    if EMB_MATRIX is None:
        return MatchResponse(top=[], best=None, count=0, dim=None, elapsed_ms=(time.time()-t0)*1000)

    # Cosine similarity = dot(q, M.T) since both normalized
    q_arr = np.asarray(q_vec, dtype=np.float32)
    sims = EMB_MATRIX @ q_arr
    # Top-k indices
    k = min(top_k, sims.shape[0])
    idx = np.argpartition(-sims, k-1)[:k]
    top_pairs = sorted(((int(i), float(sims[i])) for i in idx), key=lambda x: -x[1])

    id_to_row = {c.card_id: c for c in CARD_CACHE}
    results: List[MatchResult] = []
    for i, score in top_pairs:
        card_id = CARD_ID_INDEX[i]
        row = id_to_row[card_id]
        results.append(MatchResult(
            card_id=card_id,
            name=row.name,
            set_name=row.set_name,
            ext_number=row.ext_number,
            subtype_name=row.subtype_name,
            score=score,
        ))

    elapsed = (time.time() - t0) * 1000
    return MatchResponse(
        top=results,
        best=(results[0] if results else None),
        count=0 if EMB_MATRIX is None else EMB_MATRIX.shape[0],
        dim=None if EMB_MATRIX is None else EMB_MATRIX.shape[1],
        elapsed_ms=elapsed,
    )

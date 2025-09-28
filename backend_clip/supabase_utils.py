from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from supabase import create_client, Client
import requests
import io
from PIL import Image

from config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE, SET_FILTER, EMBED_COL

@dataclass
class CardRow:
    card_id: str
    name: str
    set_name: str
    ext_number: str
    subtype_name: str
    image_path: str
    embedding: Optional[List[float]]


def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_cards(missing_only: bool = False, limit: Optional[int] = None) -> List[CardRow]:
    sb = get_client()
    sel = "card_id,name,set_name,ext_number,subtype_name,image_path," + EMBED_COL
    q = sb.table(SUPABASE_TABLE).select(sel)
    if SET_FILTER:
        q = q.eq("set_name", SET_FILTER)
    if missing_only:
        q = q.is_(EMBED_COL, "null")
    if limit:
        q = q.limit(limit)
    data = q.execute().data or []
    rows: List[CardRow] = []
    for r in data:
        rows.append(CardRow(
            card_id=r["card_id"],
            name=r["name"],
            set_name=r["set_name"],
            ext_number=r["ext_number"],
            subtype_name=r["subtype_name"],
            image_path=r["image_path"],
            embedding=r.get(EMBED_COL),
        ))
    return rows


def update_embedding(card_id: str, vec: List[float]):
    sb = get_client()
    sb.table(SUPABASE_TABLE).update({EMBED_COL: vec}).eq("card_id", card_id).execute()


def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

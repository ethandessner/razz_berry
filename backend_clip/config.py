import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "destinedrivals")
SET_FILTER = os.getenv("SET_FILTER") or None
EMBED_COL = os.getenv("EMBED_COL", "embedding_vec")
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_EMBED_MODEL = os.getenv("VERTEX_EMBED_MODEL", "multimodalembedding@001")
TOPK = int(os.getenv("TOPK", "5"))

# Safety checks (not throwing hard errors so scripts can decide behavior)
REQUIRED = ["SUPABASE_URL", "SUPABASE_KEY", "VERTEX_PROJECT_ID"]
missing = [k for k,v in [("SUPABASE_URL", SUPABASE_URL), ("SUPABASE_KEY", SUPABASE_KEY), ("VERTEX_PROJECT_ID", VERTEX_PROJECT_ID)] if not v]
if missing:
    print(f"[config] WARNING missing env vars: {missing}")

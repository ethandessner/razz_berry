#!/usr/bin/env python3
"""Batch embedding generator for all card images in a GCS set folder.

Workflow:
  1. Lists objects in gs://<bucket>/<set_name>/ matching pattern <card_id>_<variant>.png
  2. For each object, builds a gs:// URI and fetches a multimodal image embedding via Vertex AI.
  3. L2 normalizes the embedding vector.
  4. Upserts (card_id, subtype_name) rows into Supabase, adding / updating the embedding column.

Assumptions:
  - Supabase table already contains rows for (card_id, subtype_name) OR you pass --create-missing.
  - Filenames follow: <card_id>_<variant>.png (variant lower-case, underscores allowed).
  - Environment variables provide credentials: SUPABASE_URL, SUPABASE_KEY, VERTEX_PROJECT_ID.

Example:
  python batch_embed_gcs.py \
      --set-name SV10DestinedRivals \
      --bucket razz_berry \
      --table destinedrivals \
      --dimension 512 \
      --batch-size 80 \
      --skip-existing

Flags:
  --skip-existing   Skip rows that already have a non-null embedding.
  --overwrite       Force recomputation even if embedding exists (cannot use with --skip-existing).
  --create-missing  Insert a minimal row if (card_id, variant) not found yet.
  --limit N         Only process first N objects (debug/testing).

Exit codes:
  0 success (even with some per-image failures; see summary)
  1 configuration / env error
  2 Vertex init failure
"""
from __future__ import annotations
import os, re, sys, time, math, json, argparse
from typing import List, Optional, Dict, Tuple


from dotenv import load_dotenv
load_dotenv()

try:
    import vertexai
    from vertexai.vision_models import Image, MultiModalEmbeddingModel
except ImportError as e:
    print("[fatal] vertexai SDK not installed. pip install --upgrade google-cloud-aiplatform", file=sys.stderr)
    sys.exit(1)

try:
    from google.cloud import storage
except ImportError:
    print("[fatal] google-cloud-storage not installed. pip install google-cloud-storage", file=sys.stderr)
    sys.exit(1)

try:
    from supabase import create_client
except ImportError:
    print("[fatal] supabase client not installed. pip install supabase", file=sys.stderr)
    sys.exit(1)

import numpy as np

# ------------ Helpers -------------
FILENAME_RE = re.compile(r"^(?P<card>[^_]+)_(?P<variant>[^.]+)\.png$")


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=float)
    norm = float(np.linalg.norm(arr) or 1.0)
    return (arr / norm).astype(float).tolist()


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Batch embed all card images in a GCS set prefix")
    p.add_argument("--set-name", default="SV10DestinedRivals")
    p.add_argument("--bucket", default=os.getenv("GCS_BUCKET_NAME", "razz_berry"))
    p.add_argument("--project", default=os.getenv("VERTEX_PROJECT_ID", ""))
    p.add_argument("--location", default=os.getenv("VERTEX_LOCATION", "us-central1"))
    p.add_argument("--model", default=os.getenv("VERTEX_EMBED_MODEL", "multimodalembedding@001"))
    p.add_argument("--table", default=os.getenv("SUPABASE_TABLE", "destinedrivals"))
    p.add_argument("--embed-col", default=os.getenv("EMBEDDINGS_COLUMN", "embedding_vec"))
    p.add_argument("--dimension", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=80)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--create-missing", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="List planned operations without calling Vertex or DB")
    return p.parse_args(argv)


# ------------ Core logic -------------

def init_supabase() -> Optional[object]:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        print("[fatal] SUPABASE_URL / SUPABASE_KEY missing")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        print(f"[fatal] Supabase init failed: {e}")
        return None


def init_vertex(project: str, location: str):
    try:
        vertexai.init(project=project, location=location)
        return True
    except Exception as e:
        print(f"[fatal] Vertex init failed: {e}")
        return False


def list_gcs_objects(bucket: str, prefix: str) -> List[str]:
    client = storage.Client()
    blobs = client.list_blobs(bucket, prefix=prefix)
    out = []
    for b in blobs:
        name = b.name
        # We only want direct png files under the prefix, skip folders or nested directories if not matching
        if not name.lower().endswith('.png'):
            continue
        # enforce that prefix is at start and there's at least something after
        if not name.startswith(prefix.rstrip('/') + '/'):
            continue
        fname = name.split('/')[-1]
        if FILENAME_RE.match(fname):
            out.append(name)
        else:
            print(f"[warn] skipping non-matching filename: {name}")
    return out


def fetch_existing_embeddings_map(supabase, table: str, embed_col: str, card_variant_pairs: List[Tuple[str,str]]) -> Dict[Tuple[str,str], bool]:
    # Because Supabase free tier may limit large IN queries, do small batches
    presence = {}
    B = 200
    for i in range(0, len(card_variant_pairs), B):
        chunk = card_variant_pairs[i:i+B]
        # Build a filter expression (Supabase Python client doesn't have composite key filter easily; we use OR)
        # Fallback approach: fetch by card_id IN and then filter client-side.
        card_ids = list({cv[0] for cv in chunk})
        try:
            resp = supabase.table(table).select("card_id, subtype_name, {}".format(embed_col)).in_("card_id", card_ids).execute()
            for r in resp.data or []:
                key = (r.get("card_id"), r.get("subtype_name"))
                if key in card_variant_pairs:
                    presence[key] = r.get(embed_col) is not None
        except Exception as e:
            print(f"[warn] fetch existing failed batch {i//B}: {e}")
    return presence


def derive_card_variant(obj_name: str) -> Optional[Tuple[str,str]]:
    fname = obj_name.split('/')[-1]
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    card_id = m.group('card')
    variant = m.group('variant')
    return card_id, variant


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    if args.skip_existing and args.overwrite:
        print("[fatal] --skip-existing and --overwrite are mutually exclusive")
        return 1
    if not args.project:
        print("[fatal] --project / VERTEX_PROJECT_ID required")
        return 1

    supabase = init_supabase()
    if not supabase:
        return 1

    print(f"[info] Listing GCS objects: bucket={args.bucket} prefix={args.set_name}/")
    obj_names = list_gcs_objects(args.bucket, args.set_name.rstrip('/'))
    if args.limit is not None:
        obj_names = obj_names[:args.limit]
    print(f"[info] Found {len(obj_names)} candidate PNG objects")

    # Derive card/variant pairs
    cv_pairs: List[Tuple[str,str]] = []
    for name in obj_names:
        cv = derive_card_variant(name)
        if cv:
            cv_pairs.append(cv)
    print(f"[info] Parsed {len(cv_pairs)} (card_id, variant) pairs")

    existing_map = {}
    if args.skip_existing or not args.overwrite:
        existing_map = fetch_existing_embeddings_map(supabase, args.table, args.embed_col, cv_pairs)
        print(f"[info] Existing rows fetched: {len(existing_map)}")

    if args.dry_run:
        planned = 0
        for cv in cv_pairs:
            has = existing_map.get(cv, False)
            if args.skip_existing and has:
                continue
            planned += 1
        print(f"[dry-run] Would embed {planned} of {len(cv_pairs)} images")
        return 0

    if not init_vertex(args.project, args.location):
        return 2

    model = MultiModalEmbeddingModel.from_pretrained(args.model)

    # Stats
    total = 0
    attempted = 0
    success = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    batch_records: List[dict] = []
    B = args.batch_size

    for idx, obj_name in enumerate(obj_names, 1):
        cv = derive_card_variant(obj_name)
        if not cv:
            continue
        card_id, variant = cv
        has_embed = existing_map.get(cv, False)
        if args.skip_existing and has_embed:
            skipped += 1
            continue
        if (not args.overwrite) and has_embed:
            skipped += 1
            continue

        gs_uri = f"gs://{args.bucket}/{obj_name}"
        total += 1
        attempted += 1
        try:
            image = Image.load_from_file(gs_uri)
            emb = model.get_embeddings(image=image, dimension=args.dimension)
            vec = list(emb.image_embedding)
            vec = l2_normalize(vec)
            batch_records.append({
                "card_id": card_id,
                "subtype_name": variant,
                args.embed_col: vec,
            })
            success += 1
        except Exception as e:
            failed += 1
            print(f"[error] embed failed {card_id} {variant}: {e}")
            continue

        if len(batch_records) >= B:
            try:
                supabase.table(args.table).upsert(batch_records, on_conflict="card_id,subtype_name").execute()
                print(f"[batch] upserted {len(batch_records)} (progress {idx}/{len(obj_names)})")
            except Exception as e:
                print(f"[warn] batch upsert error: {e}")
            batch_records = []

    if batch_records:
        try:
            supabase.table(args.table).upsert(batch_records, on_conflict="card_id,subtype_name").execute()
            print(f"[final] upserted {len(batch_records)}")
        except Exception as e:
            print(f"[warn] final upsert error: {e}")

    elapsed = time.time() - start_time
    rate = success / elapsed if elapsed else 0.0
    print("\n==== SUMMARY ====")
    print(f"total_objects_scanned={len(obj_names)}")
    print(f"embeddings_attempted={attempted}")
    print(f"success={success}")
    print(f"failed={failed}")
    print(f"skipped={skipped}")
    print(f"elapsed_sec={elapsed:.1f}")
    print(f"success_rate_per_sec={rate:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

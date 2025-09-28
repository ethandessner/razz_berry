#!/usr/bin/env python
"""Populate missing embeddings for cards.

Usage:
  python ingest_embeddings.py --limit 500

Idempotent: skips cards that already have embedding_vec unless --force provided.
"""
from __future__ import annotations
import argparse, time
from supabase_utils import fetch_cards, update_embedding, download_image
from vertex_embed import embed_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='Limit number of cards fetched')
    ap.add_argument('--force', action='store_true', help='Recompute even if embedding exists')
    args = ap.parse_args()

    cards = fetch_cards(missing_only=not args.force, limit=args.limit)
    print(f"Fetched {len(cards)} cards for embedding (force={args.force})")

    success, fail = 0, 0
    for i, card in enumerate(cards, 1):
        try:
            img = download_image(card.image_path)
            vec = embed_image(img)
            update_embedding(card.card_id, vec)
            success += 1
            print(f"[{i}/{len(cards)}] {card.card_id} OK (dim={len(vec)})")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(cards)}] {card.card_id} FAIL: {e}")
        time.sleep(0.1)  # mild pacing to avoid burst rate limits

    print(f"Done. success={success} fail={fail}")


if __name__ == '__main__':
    main()

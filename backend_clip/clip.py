#!/usr/bin/env python3
"""Utility script to fetch a multimodal embedding (image + optional text) from Vertex AI.

Supports image sources:
  - gs://bucket/path/to/file.png
  - https://... (public URL; downloaded to bytes)
  - Local filesystem path

Example:
  python clip.py \
      --project cardconnect-ethandessner \
      --location us-central1 \
      --model multimodalembedding@001 \
      --image gs://razz_berry/SV10DestinedRivals/sv10-122_holo.png \
      --text "Colosseum" \
      --dimension 1408

Outputs vector lengths and first few dimensions; optionally writes JSON if --output-json specified.
"""
import os
import sys
import math
import json
import argparse
import urllib.request
from typing import Optional

import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel


def load_image_any(path: str) -> Image:
    """Return a vertexai Image from gs://, http(s) URL, or local path."""
    if path.startswith("gs://"):
        # SDK understands GCS path directly
        return Image.load_from_file(path)
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path) as resp:
            data = resp.read()
        return Image(image_bytes=data)
    # Local file
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.load_from_file(path)


def l2_normalize(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Fetch multimodal embeddings from Vertex AI")
    p.add_argument("--project", default=os.getenv("VERTEX_PROJECT_ID", "cardconnect-ethandessner"))
    p.add_argument("--location", default=os.getenv("VERTEX_LOCATION", "us-central1"))
    p.add_argument("--model", default=os.getenv("VERTEX_EMBED_MODEL", "multimodalembedding@001"))
    p.add_argument("--image", required=True, help="gs://, http(s) URL, or local path")
    p.add_argument("--text", default=None, help="Optional contextual text for multimodal embedding")
    p.add_argument("--dimension", type=int, default=1408, help="Target embedding dimension if model supports")
    p.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization of vectors")
    p.add_argument("--output-json", default=None, help="Path to write embeddings JSON {image: [...], text: [...]} ")
    return p.parse_args(argv)


def main(argv: Optional[list] = None):
    args = parse_args(argv)

    print(f"[init] project={args.project} location={args.location} model={args.model}")
    vertexai.init(project=args.project, location=args.location)

    model = MultiModalEmbeddingModel.from_pretrained(args.model)

    try:
        image_obj = load_image_any(args.image)
    except Exception as e:
        print(f"[error] Failed to load image: {e}", file=sys.stderr)
        sys.exit(1)

    embed_kwargs = {
        "image": image_obj,
        "dimension": args.dimension,
    }
    if args.text:
        embed_kwargs["contextual_text"] = args.text

    try:
        embeddings = model.get_embeddings(**embed_kwargs)
    except Exception as e:
        print(f"[error] Embedding request failed: {e}", file=sys.stderr)
        sys.exit(2)

    img_vec = list(embeddings.image_embedding)
    txt_vec = list(embeddings.text_embedding) if embeddings.text_embedding else []

    if not args.no_normalize:
        img_vec = l2_normalize(img_vec)
        if txt_vec:
            txt_vec = l2_normalize(txt_vec)

    print(f"Image embedding length: {len(img_vec)} (normalized={not args.no_normalize})")
    print("First 80 image dims:", img_vec[:80])
    if txt_vec:
        print(f"Text embedding length: {len(txt_vec)}")
        print("First 8 text dims:", txt_vec[:8])
    else:
        if args.text:
            print("[warn] No text embedding returned (model/version may not include it)")

    if args.output_json:
        payload = {"image": img_vec, "text": txt_vec}
        try:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            print(f"Saved embeddings to {args.output_json}")
        except Exception as e:
            print(f"[warn] Failed to write JSON: {e}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
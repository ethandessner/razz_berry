# Embedding-Based Backend (CLIP via Vertex AI Multimodal Embeddings)

This backend is an *experimental* alternative to the existing hash-based matcher (left untouched in `backend/`). It uses Google Vertex AI's **Multimodal Embeddings** model (CLIP-like) to embed trading card images and perform nearest-neighbor similarity search.

## Overview

Flow:
1. Ingest all card reference images from Supabase table.
2. Generate an embedding vector per card using Vertex AI.
3. Store embedding back into Supabase (new column `embedding_vec` as JSON list of floats or as a base64/array depending on preference).
4. At query time: receive uploaded image, embed it, perform vector similarity against cached embeddings (kept in RAM), return top-k matches.

## Components

- `config.py` – Loads environment variables.
- `supabase_utils.py` – Fetch & update card rows in Supabase.
- `vertex_embed.py` – Wrapper for Vertex AI embedding calls.
- `ingest_embeddings.py` – Batch script to populate missing embeddings.
- `server.py` – FastAPI server with `/health` and `/match` endpoints using embeddings.
- `requirements.txt` – Python deps.

## Environment Variables
Create `.env` (not committed) with:
```
SUPABASE_URL=...
SUPABASE_KEY=...
SUPABASE_TABLE=destinedrivals
SET_FILTER=
EMBED_COL=embedding_vec
VERTEX_PROJECT_ID=your-gcp-project
VERTEX_LOCATION=us-central1
# One of the available models per Vertex docs (example below):
VERTEX_EMBED_MODEL=multimodalembedding@001
TOPK=5
```
Make sure your environment is authenticated for Vertex AI (e.g., `gcloud auth application-default login` OR provide a service account JSON via `GOOGLE_APPLICATION_CREDENTIALS`).

## Supabase Schema Change
Add a new column (e.g. `embedding_vec`) of type `jsonb` (recommended) to store an array of floats:
```sql
ALTER TABLE destinedrivals ADD COLUMN embedding_vec jsonb;
-- optional index (GIN) for containment queries, though we handle vector search in memory here
```
If using pgvector extension (better similarity search inside DB):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE destinedrivals ADD COLUMN embedding_vec vector(1408); -- if model dimension is 1408 (example)
```
(Confirm the actual dimension returned by the selected model; adjust accordingly.)

## Running Ingestion
```
python ingest_embeddings.py --limit 500
```
This will:
- Fetch cards lacking `embedding_vec`.
- Download each `image_path` (assumes it's a URL) or fetch from storage.
- Call Vertex embedding API.
- Write embedding back to Supabase.

## Running Server
```
uvicorn server:app --reload --port 8081
```
Then query:
```
curl -F file=@sample.jpg http://localhost:8081/match
```

## Matching Strategy
- Cosine similarity between normalized embedding vectors.
- All card embeddings cached in RAM at startup; refresh endpoint TBD.
- For large scale (10k+), consider FAISS, Annoy, or pgvector in-DB search.

## Roadmap / TODO
- Add periodic refresh endpoint.
- Add batch inference optimization (send up to 16 images per API call if supported).
- Fall back to hash engine if embedding confidence low.
- Confidence scoring calibration.

## Disclaimer
This is experimental; costs accrue per embedding call. Cache aggressively and avoid recomputing unchanged images.

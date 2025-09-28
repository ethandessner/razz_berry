from __future__ import annotations
from typing import List
import numpy as np
from google.cloud import aiplatform
# from google.cloud.aiplatform.gapic.schema import predict
from config import VERTEX_PROJECT_ID, VERTEX_LOCATION, VERTEX_EMBED_MODEL
import base64
import io
from PIL import Image

# Initialize Vertex client lazily
_client_initialized = False


def _init_client():
    global _client_initialized
    if _client_initialized:
        return
    aiplatform.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
    _client_initialized = True


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def embed_image(img: Image.Image) -> List[float]:
    """Return the normalized embedding vector for the image using Vertex multimodal embeddings.
    Model reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api
    """
    _init_client()
    from google.cloud.aiplatform import PredictionServiceClient
    client = PredictionServiceClient()

    endpoint = f"projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_LOCATION}/publishers/google/models/{VERTEX_EMBED_MODEL}"

    img_bytes = pil_to_jpeg_bytes(img, quality=90)
    b64 = base64.b64encode(img_bytes).decode()

    instance = {
        "image": {"bytesBase64Encoded": b64},
        # Optionally specify modality-specific params here
    }

    instances = [instance]
    parameters = {}

    resp = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    # Response structure: predictions[0]['imageEmbedding'] assumed (per model docs)
    pred = resp.predictions[0]
    # Different SDK versions may wrap in Value proto; handle both
    if hasattr(pred, 'get'):
        embed = pred.get('imageEmbedding') or pred.get('embedding')
    else:
        # Fallback: attempt attribute access / sequence
        embed = pred['imageEmbedding'] if 'imageEmbedding' in pred else pred['embedding']
    vec = list(map(float, embed))
    # Normalize to unit length for cosine similarity
    norm = np.linalg.norm(vec) or 1.0
    return [float(x / norm) for x in vec]

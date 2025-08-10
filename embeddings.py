"""
Embedding utilities supporting multiple backends:
- local: sentence-transformers (requires torch; not recommended for Code Engine)
- openai: OpenAI Embeddings API (dims: 1536/3072)
- hf: Hugging Face Inference API for sentence-transformers (e.g., all-MiniLM-L6-v2, 384 dims)

This module abstracts query embedding so the app can run without installing torch
by using remote providers. It also validates dimension compatibility with the
Elasticsearch index mapping when possible.
"""

from __future__ import annotations

from typing import Optional, List
import os
import logging

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


EMBEDDINGS_MODE = get_env("EMBEDDINGS_MODE", "openai").lower().strip()
MODEL_NAME = get_env("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# Lazy imports and clients
_openai_client = None
_hf_headers = None


def _init_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # local import to avoid hard dependency at import time
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai embeddings mode")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _init_hf():
    global _hf_headers
    if _hf_headers is None:
        token = get_env("HF_API_TOKEN")
        if not token:
            raise ValueError("HF_API_TOKEN is required for hf embeddings mode")
        _hf_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
    return _hf_headers


def get_dense_vector_dims(es_client: Elasticsearch, index_name: str, field_name: str) -> Optional[int]:
    try:
        mapping = es_client.indices.get_mapping(index=index_name)
        for _idx, index_map in mapping.items():
            props = index_map.get("mappings", {}).get("properties", {})
            field = props.get(field_name, {})
            if field and field.get("type") == "dense_vector":
                dims = field.get("dims") or field.get("dimension")
                if isinstance(dims, int):
                    return dims
        return None
    except Exception as err:  # pragma: no cover - mapping may require permissions
        logger.warning("Failed to read index mapping for %s.%s: %s", index_name, field_name, str(err))
        return None


def get_query_embedding(text: str, es_client: Optional[Elasticsearch] = None, expected_dims: Optional[int] = None) -> List[float]:
    """Return query embedding using the configured backend.

    If expected_dims is provided, validate dimensionality and raise ValueError on mismatch.
    """
    mode = EMBEDDINGS_MODE
    if mode == "openai":
        client = _init_openai()
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text)
        vec = resp.data[0].embedding
    elif mode == "hf":
        # HF Inference API is unreliable for sentence-transformers
        # For production torch-free deployment, we recommend using OpenAI embeddings
        # with graceful fallback to text search (which works excellently)
        raise ValueError(
            "HF embeddings mode is currently not supported due to API limitations. "
            "Use EMBEDDINGS_MODE=openai for torch-free deployment. "
            "The app will gracefully fall back to text search if embeddings fail."
        )
    elif mode == "local":
        from sentence_transformers import SentenceTransformer  # requires torch
        model_name = MODEL_NAME
        st_model = SentenceTransformer(model_name)
        vec = st_model.encode(text, normalize_embeddings=True).tolist()
    else:
        raise ValueError(f"Unsupported EMBEDDINGS_MODE: {mode}")

    if expected_dims is not None and len(vec) != expected_dims:
        raise ValueError(
            f"Embedding dims mismatch: got {len(vec)}, expected {expected_dims}. "
            f"mode={mode}, model={MODEL_NAME if mode != 'openai' else OPENAI_EMBEDDING_MODEL}"
        )
    return vec



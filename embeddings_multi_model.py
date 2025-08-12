# Enhanced Multi-Model Embeddings Module
# Supports multiple embedding models for search operations

import os
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv, find_dotenv
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv(find_dotenv(), override=False)

# Embedding configuration
EMBEDDINGS_MODE = os.getenv("EMBEDDINGS_MODE", "local")  # openai, hf, local
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Multi-model configuration - matches upload script
EMBEDDING_MODELS = {
    "v1": {
        "type": "sentence_transformers",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dims": 384,
        "vector_field": "body_vector"
    },
    "v2": {
        "type": "openai", 
        "model_name": "text-embedding-3-small",
        "dims": 1536,
        "vector_field": "body_vector_v2"
    },
    "v3": {
        "type": "sentence_transformers",
        "model_name": "sentence-transformers/all-mpnet-base-v2", 
        "dims": 768,
        "vector_field": "body_vector_v3"
    }
}

# Default model for search (can be overridden)
DEFAULT_SEARCH_MODEL = os.getenv("DEFAULT_SEARCH_MODEL", "v1")

# Initialize clients
_openai_client = None
_st_models = {}


def _init_openai():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            _openai_client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    return _openai_client


def _init_sentence_transformer(model_name: str):
    if model_name not in _st_models:
        try:
            from sentence_transformers import SentenceTransformer
            _st_models[model_name] = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Run: pip install sentence-transformers")
    return _st_models[model_name]


def get_available_models(es_client: Elasticsearch, index_name: str) -> List[str]:
    """Get list of available embedding models in the index."""
    try:
        mapping = es_client.indices.get_mapping(index=index_name)
        available_models = []
        
        for _idx, index_map in mapping.items():
            props = index_map.get("mappings", {}).get("properties", {})
            
            for model_key, config in EMBEDDING_MODELS.items():
                vector_field = config["vector_field"]
                if vector_field in props and props[vector_field].get("type") == "dense_vector":
                    available_models.append(model_key)
        
        return available_models
    except Exception as e:
        print(f"Could not determine available models: {e}")
        return ["v1"]  # Fallback to default


def get_dense_vector_dims(es_client: Elasticsearch, index_name: str, field_name: str) -> Optional[int]:
    """Get dimensions of a dense vector field."""
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
    except Exception as err:
        print(f"Could not get vector dims for {index_name}.{field_name}: {err}")
        return None


def get_query_embedding(text: str, model_key: str = None, es_client: Optional[Elasticsearch] = None) -> List[float]:
    """Get query embedding using specified model key."""
    if model_key is None:
        model_key = DEFAULT_SEARCH_MODEL
    
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(EMBEDDING_MODELS.keys())}")
    
    model_config = EMBEDDING_MODELS[model_key]
    
    if model_config["type"] == "openai":
        client = _init_openai()
        resp = client.embeddings.create(model=model_config["model_name"], input=text)
        return resp.data[0].embedding
    
    elif model_config["type"] == "sentence_transformers":
        model = _init_sentence_transformer(model_config["model_name"])
        return model.encode(text, normalize_embeddings=True).tolist()
    
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")


def get_query_embedding_legacy(text: str, es_client: Optional[Elasticsearch] = None, expected_dims: Optional[int] = None) -> List[float]:
    """Legacy function for backward compatibility."""
    mode = EMBEDDINGS_MODE
    if mode == "openai":
        client = _init_openai()
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text)
        vec = resp.data[0].embedding
    elif mode == "hf":
        raise ValueError(
            "HF embeddings mode is currently not supported due to API limitations. "
            "Use EMBEDDINGS_MODE=openai for torch-free deployment. "
            "The app will gracefully fall back to text search if embeddings fail."
        )
    elif mode == "local":
        model = _init_sentence_transformer(MODEL_NAME)
        vec = model.encode(text, normalize_embeddings=True).tolist()
    else:
        raise ValueError(f"Unsupported EMBEDDINGS_MODE: {mode}")

    if expected_dims is not None and len(vec) != expected_dims:
        raise ValueError(
            f"Embedding dims mismatch: got {len(vec)}, expected {expected_dims}. "
            f"mode={mode}, model={MODEL_NAME if mode != 'openai' else OPENAI_EMBEDDING_MODEL}"
        )
    return vec


def build_vector_search_query(
    query_vector: List[float], 
    vector_field: str, 
    size: int = 10,
    min_score: float = 0.0
) -> Dict[str, Any]:
    """Build Elasticsearch vector search query."""
    return {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, params.vector_field) + 1.0",
                    "params": {
                        "query_vector": query_vector,
                        "vector_field": vector_field
                    }
                },
                "min_score": min_score + 1.0  # Adjust for +1.0 offset
            }
        },
        "size": size,
        "_source": {"excludes": [vector_field]}  # Exclude vector from results
    }


def search_with_model(
    es_client: Elasticsearch,
    index_name: str, 
    query_text: str,
    model_key: str = None,
    size: int = 10,
    min_score: float = 0.0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Search using specific embedding model."""
    
    if model_key is None:
        model_key = DEFAULT_SEARCH_MODEL
    
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    model_config = EMBEDDING_MODELS[model_key]
    vector_field = model_config["vector_field"]
    
    # Check if model is available in index
    available_models = get_available_models(es_client, index_name)
    if model_key not in available_models:
        raise ValueError(f"Model {model_key} not available in index {index_name}. Available: {available_models}")
    
    # Generate query embedding
    query_vector = get_query_embedding(query_text, model_key)
    
    # Build and execute search
    search_query = build_vector_search_query(query_vector, vector_field, size, min_score)
    response = es_client.search(index=index_name, **search_query)
    
    # Extract results
    results = []
    for hit in response["hits"]["hits"]:
        result = {
            "document_id": hit["_id"],
            "similarity_score": hit["_score"] - 1.0,  # Remove +1.0 offset
            "source": hit["_source"]
        }
        results.append(result)
    
    metadata = {
        "model_used": model_key,
        "vector_field": vector_field,
        "total_hits": response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else response["hits"]["total"],
        "max_score": response["hits"]["max_score"] - 1.0 if response["hits"]["max_score"] else 0.0
    }
    
    return results, metadata


def search_with_multiple_models(
    es_client: Elasticsearch,
    index_name: str,
    query_text: str, 
    model_keys: List[str] = None,
    size: int = 10,
    ensemble_method: str = "average"  # "average", "max", "weighted"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Search using multiple models and combine results."""
    
    if model_keys is None:
        model_keys = get_available_models(es_client, index_name)
    
    all_results = {}  # doc_id -> {scores: [], source: dict}
    model_metadata = {}
    
    # Search with each model
    for model_key in model_keys:
        try:
            results, metadata = search_with_model(es_client, index_name, query_text, model_key, size * 2)  # Get more results for ensemble
            model_metadata[model_key] = metadata
            
            for result in results:
                doc_id = result["document_id"]
                if doc_id not in all_results:
                    all_results[doc_id] = {"scores": [], "source": result["source"]}
                all_results[doc_id]["scores"].append(result["similarity_score"])
        
        except Exception as e:
            print(f"Failed to search with model {model_key}: {e}")
            model_metadata[model_key] = {"error": str(e)}
    
    # Combine scores using ensemble method
    final_results = []
    for doc_id, data in all_results.items():
        scores = data["scores"]
        
        if ensemble_method == "average":
            combined_score = sum(scores) / len(scores)
        elif ensemble_method == "max":
            combined_score = max(scores)
        elif ensemble_method == "weighted":
            # Weight by model performance (could be more sophisticated)
            weights = [1.0] * len(scores)  # Equal weights for now
            combined_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            combined_score = sum(scores) / len(scores)  # Default to average
        
        final_results.append({
            "document_id": doc_id,
            "similarity_score": combined_score,
            "individual_scores": {model_keys[i]: scores[i] for i in range(len(scores))},
            "source": data["source"]
        })
    
    # Sort by combined score and limit results
    final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    final_results = final_results[:size]
    
    metadata = {
        "ensemble_method": ensemble_method,
        "models_used": model_keys,
        "model_metadata": model_metadata,
        "total_combined_results": len(final_results)
    }
    
    return final_results, metadata


def get_model_info(model_key: str = None) -> Dict[str, Any]:
    """Get information about a specific model or all models."""
    if model_key is None:
        return {"available_models": EMBEDDING_MODELS}
    
    if model_key not in EMBEDDING_MODELS:
        return {"error": f"Unknown model key: {model_key}"}
    
    return {"model_key": model_key, **EMBEDDING_MODELS[model_key]}

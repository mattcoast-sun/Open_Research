# Enhanced Multi-Model Embedding Upload Script
# pip install elasticsearch sentence-transformers openai
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple, Optional
from dotenv import load_dotenv, find_dotenv
import openai
from openai import OpenAI

# Load .env if present
load_dotenv(find_dotenv(), override=False)


def get_first_env(names: List[str], default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


# Elasticsearch connection/config (supports multiple common variable names)
ES_URL = get_first_env(
    [
        "ES_URL",
        "ELASTICSEARCH_URL",
        "ELASTIC_URL",
        "ELASTIC_HOST",
        "ELASTICSEARCH_HOST",
        "ELASTICSEARCH_HOSTS",
    ],
    "http://localhost:9200",
)
ES_USER = get_first_env([
    "ES_USER",
    "ELASTICSEARCH_USERNAME",
    "ELASTIC_USER",
    "ELASTIC_USERNAME",
]) or "elastic"
ES_PASS = get_first_env([
    "ES_PASS",
    "ELASTICSEARCH_PASSWORD",
    "ELASTIC_PASSWORD",
    "ELASTIC_PASS",
]) or "your_password"
ES_PORT = get_first_env(["ES_PORT", "ELASTICSEARCH_PORT", "ELASTIC_PORT"], "").strip()

# Elastic Cloud optional configuration
CLOUD_ID = get_first_env(["CLOUD_ID", "ELASTIC_CLOUD_ID"]) or None
API_KEY = get_first_env(["ES_API_KEY", "ELASTIC_API_KEY", "API_KEY"]) or None

# Defaults to this workspace's JSON files
DEFAULT_CLOUDS_PATH = "/Users/mattiasacosta/Documents/Programming_Projects/Open_Research/clouds.json"
DEFAULT_BEST_PRACTICES_PATH = "/Users/mattiasacosta/Documents/Programming_Projects/Open_Research/cloud_best_practices.json"

CLOUDS_JSON_PATH = os.getenv("JSON_PATH_1", DEFAULT_CLOUDS_PATH)
BEST_PRACTICES_JSON_PATH = os.getenv("JSON_PATH_2", DEFAULT_BEST_PRACTICES_PATH)

# Two dedicated indices
BASE_INDEX = os.getenv("INDEX", "cloud_providers")
CLOUDS_INDEX = f"{BASE_INDEX}_providers" if BASE_INDEX != "providers" else "cloud_providers"
BEST_PRACTICES_INDEX = f"{BASE_INDEX}_best_practices" if BASE_INDEX != "providers" else "cloud_best_practices"

# Multi-Model Configuration
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

# Which models to use for this upload (comma-separated)
ACTIVE_MODELS = os.getenv("ACTIVE_MODELS", "v1").split(",")

# Whether to update existing documents vs create new ones
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "true").lower() in {"1", "true", "yes"}

# Whether to delete and recreate indices (dangerous!)
RECREATE_INDICES = os.getenv("RECREATE_INDICES", "false").lower() in {"1", "true", "yes"}

# Initialize clients
if ES_URL:
    ES_URL = ES_URL.rstrip("/")

if CLOUD_ID and API_KEY:
    es = Elasticsearch(cloud_id=CLOUD_ID, api_key=API_KEY)
else:
    if ES_PORT and ":" not in ES_URL.split("//", 1)[-1].split("/", 1)[0]:
        ES_URL = f"{ES_URL}:{ES_PORT}"
    es = Elasticsearch(
        ES_URL, 
        http_auth=(ES_USER, ES_PASS),
        verify_certs=False,
        use_ssl=True
    )

# Initialize OpenAI if needed
openai_client = None
if any(EMBEDDING_MODELS[m]["type"] == "openai" for m in ACTIVE_MODELS):
    openai_api_key = get_first_env(["OPENAI_API_KEY"])
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        print("Warning: OpenAI model requested but OPENAI_API_KEY not found")

# Initialize sentence transformer models
st_models = {}
for model_key in ACTIVE_MODELS:
    model_config = EMBEDDING_MODELS[model_key]
    if model_config["type"] == "sentence_transformers":
        model_name = model_config["model_name"]
        if model_name not in st_models:
            st_models[model_name] = SentenceTransformer(model_name)


def get_embedding(text: str, model_key: str) -> List[float]:
    """Get embedding for text using specified model."""
    model_config = EMBEDDING_MODELS[model_key]
    
    if model_config["type"] == "sentence_transformers":
        model = st_models[model_config["model_name"]]
        return model.encode(text, normalize_embeddings=True).tolist()
    
    elif model_config["type"] == "openai":
        if not openai_client:
            raise ValueError(f"OpenAI client not initialized for model {model_key}")
        
        response = openai_client.embeddings.create(
            model=model_config["model_name"],
            input=text
        )
        return response.data[0].embedding
    
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")


def get_existing_document(index_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """Get existing document from Elasticsearch."""
    try:
        response = es.get(index=index_name, id=doc_id)
        return response["_source"]
    except Exception:
        return None


def ensure_index_with_multi_vectors(index_name: str, base_mapping: Dict[str, Any]) -> None:
    """Create or update index to support multiple vector fields."""
    exists = es.indices.exists(index=index_name)
    
    if exists and RECREATE_INDICES:
        print(f"Deleting existing index: {index_name}")
        es.indices.delete(index=index_name, ignore_unavailable=True)
        exists = False
    
    # Add vector fields for all configured models
    mapping_properties = base_mapping.copy()
    for model_key, model_config in EMBEDDING_MODELS.items():
        vector_field = model_config["vector_field"] 
        mapping_properties[vector_field] = {
            "type": "dense_vector", 
            "dims": model_config["dims"]
        }
    
    if not exists:
        print(f"Creating index: {index_name}")
        es.indices.create(
            index=index_name,
            mappings={"properties": mapping_properties}
        )
    else:
        # Update mapping for new vector fields
        print(f"Updating mapping for index: {index_name}")
        try:
            es.indices.put_mapping(
                index=index_name,
                properties=mapping_properties
            )
        except Exception as e:
            print(f"Warning: Could not update mapping for {index_name}: {e}")


def read_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a top-level JSON array in {path}")
        return data


def concatenate_provider_fields(record: Dict[str, Any]) -> str:
    parts: List[str] = [
        record.get("provider_name") or "",
        record.get("overview") or "",
    ]
    for key in [
        "advantages",
        "weaknesses", 
        "pricing_structures",
        "core_features",
        "use_cases",
        "success_stories",
        "analyst_opinions",
    ]:
        value = record.get(key)
        if isinstance(value, list):
            parts.extend([str(v) for v in value])
        elif isinstance(value, str):
            parts.append(value)
    return "\n".join([p for p in parts if p])


def concatenate_best_practice_fields(record: Dict[str, Any]) -> str:
    parts: List[str] = [
        str(record.get("id", "")),
        record.get("category") or "",
        record.get("title") or "",
        record.get("summary") or "",
        record.get("detailed_description") or "",
        record.get("impact_level") or "",
        record.get("implementation_effort") or "",
    ]
    examples = record.get("examples")
    if isinstance(examples, list):
        for example_obj in examples:
            if isinstance(example_obj, dict):
                example_text = example_obj.get("example")
                if isinstance(example_text, str):
                    parts.append(example_text)
            elif isinstance(example_obj, str):
                parts.append(example_obj)
    return "\n".join([p for p in parts if p])


def build_provider_doc(record: Dict[str, Any], vectors: Dict[str, List[float]]) -> Dict[str, Any]:
    doc = {
        "provider_name": record.get("provider_name"),
        "provider_name_text": record.get("provider_name"),
        "overview": record.get("overview"),
        "advantages": record.get("advantages", []),
        "weaknesses": record.get("weaknesses", []),
        "pricing_structures": record.get("pricing_structures"),
        "core_features": record.get("core_features", []),
        "use_cases": record.get("use_cases", []),
        "success_stories": record.get("success_stories", []),
        "analyst_opinions": record.get("analyst_opinions", []),
    }
    # Add all vector fields
    doc.update(vectors)
    return doc


def build_best_practice_doc(record: Dict[str, Any], vectors: Dict[str, List[float]]) -> Dict[str, Any]:
    # Normalize examples to simple list of strings
    examples_list: List[str] = []
    examples = record.get("examples")
    if isinstance(examples, list):
        for example_obj in examples:
            if isinstance(example_obj, dict) and isinstance(example_obj.get("example"), str):
                examples_list.append(example_obj["example"])
            elif isinstance(example_obj, str):
                examples_list.append(example_obj)

    doc = {
        "id": record.get("id"),
        "category": record.get("category"),
        "title": record.get("title"),
        "summary": record.get("summary"),
        "detailed_description": record.get("detailed_description"),
        "impact_level": record.get("impact_level"),
        "implementation_effort": record.get("implementation_effort"),
        "examples": examples_list,
    }
    # Add all vector fields
    doc.update(vectors)
    return doc


def to_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") if value else ""


def index_documents_multi_model(
    index_name: str,
    records: Iterable[Dict[str, Any]],
    text_builder,
    doc_builder,
    id_builder,
) -> Tuple[int, int]:
    actions: List[Dict[str, Any]] = []
    
    for record in records:
        text_for_embedding = text_builder(record)
        if not text_for_embedding:
            continue
            
        doc_id = id_builder(record)
        
        # Check if we're updating existing document
        existing_doc = None
        if UPDATE_EXISTING:
            existing_doc = get_existing_document(index_name, doc_id)
        
        # Generate embeddings for active models
        vectors = {}
        for model_key in ACTIVE_MODELS:
            model_config = EMBEDDING_MODELS[model_key]
            vector_field = model_config["vector_field"]
            
            try:
                vector = get_embedding(text_for_embedding, model_key)
                vectors[vector_field] = vector
                print(f"Generated {model_key} embedding ({len(vector)} dims) for doc {doc_id}")
            except Exception as e:
                print(f"Failed to generate {model_key} embedding for doc {doc_id}: {e}")
                continue
        
        if not vectors:
            print(f"No vectors generated for doc {doc_id}, skipping")
            continue
        
        # Build document, preserving existing vectors if updating
        if existing_doc and UPDATE_EXISTING:
            # Preserve existing vector fields not being updated
            for model_key, model_config in EMBEDDING_MODELS.items():
                vector_field = model_config["vector_field"]
                if vector_field in existing_doc and vector_field not in vectors:
                    vectors[vector_field] = existing_doc[vector_field]
            
            # Start with existing document and update
            doc = existing_doc.copy()
            new_doc = doc_builder(record, vectors)
            doc.update(new_doc)
        else:
            # Create fresh document
            doc = doc_builder(record, vectors)
        
        actions.append({"_index": index_name, "_id": doc_id, "_source": doc})

    if not actions:
        return (0, 0)

    success, errors = helpers.bulk(es, actions, stats_only=True)
    return success, errors


def main() -> None:
    print(f"Starting multi-model upload with models: {ACTIVE_MODELS}")
    print(f"Update existing documents: {UPDATE_EXISTING}")
    print(f"Recreate indices: {RECREATE_INDICES}")
    
    # Base mappings (without vector fields)
    clouds_base_mapping = {
        "provider_name": {"type": "keyword"},
        "provider_name_text": {"type": "text"},
        "overview": {"type": "text"},
        "advantages": {"type": "text"},
        "weaknesses": {"type": "text"},
        "pricing_structures": {"type": "text"},
        "core_features": {"type": "text"},
        "use_cases": {"type": "text"},
        "success_stories": {"type": "text"},
        "analyst_opinions": {"type": "text"},
    }

    best_practices_base_mapping = {
        "id": {"type": "integer"},
        "category": {"type": "keyword"},
        "title": {"type": "text"},
        "summary": {"type": "text"},
        "detailed_description": {"type": "text"},
        "impact_level": {"type": "keyword"},
        "implementation_effort": {"type": "keyword"},
        "examples": {"type": "text"},
    }

    # Create/update indices with multi-vector support
    ensure_index_with_multi_vectors(CLOUDS_INDEX, clouds_base_mapping)
    ensure_index_with_multi_vectors(BEST_PRACTICES_INDEX, best_practices_base_mapping)

    # Load and index providers
    print(f"\nProcessing cloud providers from: {CLOUDS_JSON_PATH}")
    clouds_records = read_json_list(CLOUDS_JSON_PATH)
    clouds_success, clouds_errors = index_documents_multi_model(
        index_name=CLOUDS_INDEX,
        records=clouds_records,
        text_builder=concatenate_provider_fields,
        doc_builder=build_provider_doc,
        id_builder=lambda r: to_slug(str(r.get("provider_name", ""))) or to_slug(str(r.get("overview", ""))[:64]),
    )

    # Load and index best practices
    print(f"\nProcessing best practices from: {BEST_PRACTICES_JSON_PATH}")
    best_practices_records = read_json_list(BEST_PRACTICES_JSON_PATH)
    best_success, best_errors = index_documents_multi_model(
        index_name=BEST_PRACTICES_INDEX,
        records=best_practices_records,
        text_builder=concatenate_best_practice_fields,
        doc_builder=build_best_practice_doc,
        id_builder=lambda r: str(r.get("id") or to_slug(str(r.get("title", "")))),
    )

    print(f"\n=== UPLOAD COMPLETE ===")
    print(f"Cloud providers: {clouds_success} indexed, {clouds_errors} errors")
    print(f"Best practices: {best_success} indexed, {best_errors} errors")
    print(f"Active models: {ACTIVE_MODELS}")
    
    # Show sample document structure
    if clouds_success > 0:
        sample_id = to_slug(str(clouds_records[0].get("provider_name", "")))
        try:
            sample_doc = es.get(index=CLOUDS_INDEX, id=sample_id)
            vector_fields = [k for k in sample_doc["_source"].keys() if k.startswith("body_vector")]
            print(f"\nSample document vector fields: {vector_fields}")
        except Exception as e:
            print(f"Could not retrieve sample document: {e}")


if __name__ == "__main__":
    main()

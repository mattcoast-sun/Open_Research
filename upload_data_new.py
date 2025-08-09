# pip install elasticsearch sentence-transformers
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple
from dotenv import load_dotenv, find_dotenv

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

# Embedding model
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims

# Whether to delete and recreate indices
RECREATE_INDICES = os.getenv("RECREATE_INDICES", "true").lower() in {"1", "true", "yes"}

if ES_URL:
    ES_URL = ES_URL.rstrip("/")

if CLOUD_ID and API_KEY:
    es = Elasticsearch(cloud_id=CLOUD_ID, api_key=API_KEY)
else:
    if ES_PORT and ":" not in ES_URL.split("//", 1)[-1].split("/", 1)[0]:
        ES_URL = f"{ES_URL}:{ES_PORT}"
    es = Elasticsearch(
        ES_URL, 
        http_auth=(ES_USER, ES_PASS),  # Use http_auth instead of basic_auth for v7.x
        verify_certs=False,
        use_ssl=True
    )
model = SentenceTransformer(MODEL_NAME)


def ensure_index(index_name: str, mapping_properties: Dict[str, Any]) -> None:
    """Create a fresh index with provided mapping. Drops existing index when RECREATE_INDICES is true."""
    exists = es.indices.exists(index=index_name)
    if exists and RECREATE_INDICES:
        es.indices.delete(index=index_name, ignore_unavailable=True)
    if (not exists) or RECREATE_INDICES:
        es.indices.create(
            index=index_name,
            mappings={
                "properties": mapping_properties,
            },
        )


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


def build_provider_doc(record: Dict[str, Any], vector: List[float]) -> Dict[str, Any]:
    return {
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
        "body_vector": vector,
    }


def build_best_practice_doc(record: Dict[str, Any], vector: List[float]) -> Dict[str, Any]:
    # Normalize examples to simple list of strings
    examples_list: List[str] = []
    examples = record.get("examples")
    if isinstance(examples, list):
        for example_obj in examples:
            if isinstance(example_obj, dict) and isinstance(example_obj.get("example"), str):
                examples_list.append(example_obj["example"])
            elif isinstance(example_obj, str):
                examples_list.append(example_obj)

    return {
        "id": record.get("id"),
        "category": record.get("category"),
        "title": record.get("title"),
        "summary": record.get("summary"),
        "detailed_description": record.get("detailed_description"),
        "impact_level": record.get("impact_level"),
        "implementation_effort": record.get("implementation_effort"),
        "examples": examples_list,
        "body_vector": vector,
    }


def to_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") if value else ""


def index_documents(
    index_name: str,
    records: Iterable[Dict[str, Any]],
    text_builder,
    doc_builder,
    id_builder,
) -> Tuple[int, int]:
    actions: List[Dict[str, Any]] = []
    for record in records:
        text_for_embedding = text_builder(record)
        # Skip empty docs
        if not text_for_embedding:
            continue
        vector = model.encode(text_for_embedding, normalize_embeddings=True).tolist()
        doc = doc_builder(record, vector)
        doc_id = id_builder(record)
        actions.append({"_index": index_name, "_id": doc_id, "_source": doc})

    if not actions:
        return (0, 0)

    success, errors = helpers.bulk(es, actions, stats_only=True)
    return success, errors


def main() -> None:
    # Define simple mappings. Note: 'dense_vector' without indexing to maximize compatibility across ES versions.
    vector_mapping = {"type": "dense_vector", "dims": 384}

    clouds_mapping = {
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
        "body_vector": vector_mapping,
    }

    best_practices_mapping = {
        "id": {"type": "integer"},
        "category": {"type": "keyword"},
        "title": {"type": "text"},
        "summary": {"type": "text"},
        "detailed_description": {"type": "text"},
        "impact_level": {"type": "keyword"},
        "implementation_effort": {"type": "keyword"},
        "examples": {"type": "text"},
        "body_vector": vector_mapping,
    }

    # Create indices
    ensure_index(CLOUDS_INDEX, clouds_mapping)
    ensure_index(BEST_PRACTICES_INDEX, best_practices_mapping)

    # Load and index providers
    clouds_records = read_json_list(CLOUDS_JSON_PATH)
    clouds_success, clouds_errors = index_documents(
        index_name=CLOUDS_INDEX,
        records=clouds_records,
        text_builder=concatenate_provider_fields,
        doc_builder=build_provider_doc,
        id_builder=lambda r: to_slug(str(r.get("provider_name", ""))) or to_slug(str(r.get("overview", ""))[:64]),
    )

    # Load and index best practices
    best_practices_records = read_json_list(BEST_PRACTICES_JSON_PATH)
    best_success, best_errors = index_documents(
        index_name=BEST_PRACTICES_INDEX,
        records=best_practices_records,
        text_builder=concatenate_best_practice_fields,
        doc_builder=build_best_practice_doc,
        id_builder=lambda r: str(r.get("id") or to_slug(str(r.get("title", "")))),
    )

    print(
        f"Indexed {clouds_success} cloud provider docs into '{CLOUDS_INDEX}' (errors: {clouds_errors})."
    )
    print(
        f"Indexed {best_success} best practice docs into '{BEST_PRACTICES_INDEX}' (errors: {best_errors})."
    )


if __name__ == "__main__":
    main()


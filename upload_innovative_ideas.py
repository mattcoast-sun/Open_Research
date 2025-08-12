#!/usr/bin/env python3
"""
Custom upload script for innovative ideas and technical building blocks.
This script uploads innovation themes and technical building blocks with OpenAI embeddings.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple, Iterable
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our multi-model embedding system
from upload_data_multi_model import (
    get_first_env, EMBEDDING_MODELS, get_embedding,
    ensure_index_with_multi_vectors, get_existing_document
)

# Configuration
INNOVATIVE_IDEAS_JSON_PATH = "innovative_ideas_enhanced.json"
INNOVATIVE_IDEAS_INDEX = "innovative_ideas"

# Active models for this upload (default to OpenAI)
ACTIVE_MODELS = os.getenv("ACTIVE_MODELS", "v2").split(",")
UPDATE_EXISTING = os.getenv("UPDATE_EXISTING", "true").lower() in {"1", "true", "yes"}
RECREATE_INDICES = os.getenv("RECREATE_INDICES", "false").lower() in {"1", "true", "yes"}

# Initialize Elasticsearch
ES_URL = get_first_env([
    "ES_URL", "ELASTICSEARCH_URL", "ELASTIC_URL", "ELASTIC_HOST",
    "ELASTICSEARCH_HOST", "ELASTICSEARCH_HOSTS"
], "http://localhost:9200")
ES_USER = get_first_env([
    "ES_USER", "ELASTICSEARCH_USERNAME", "ELASTIC_USER", "ELASTIC_USERNAME"
]) or "elastic"
ES_PASS = get_first_env([
    "ES_PASS", "ELASTICSEARCH_PASSWORD", "ELASTIC_PASSWORD", "ELASTIC_PASS"
]) or "your_password"
ES_PORT = get_first_env(["ES_PORT", "ELASTICSEARCH_PORT", "ELASTIC_PORT"], "").strip()

CLOUD_ID = get_first_env(["CLOUD_ID", "ELASTIC_CLOUD_ID"]) or None
API_KEY = get_first_env(["ES_API_KEY", "ELASTIC_API_KEY", "API_KEY"]) or None

# Initialize Elasticsearch client
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

def read_json_list(path: str) -> List[Dict[str, Any]]:
    """Read JSON array from file."""
    with open(path, "r") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a top-level JSON array in {path}")
        return data

def concatenate_innovation_fields(record: Dict[str, Any]) -> str:
    """Extract and concatenate text fields for embedding generation."""
    parts: List[str] = [
        record.get("title") or "",
        record.get("summary") or "",
        record.get("detailed_description") or "",
        record.get("category") or "",
        record.get("type") or "",
        record.get("impact_level") or "",
        record.get("implementation_effort") or "",
    ]
    
    # Add examples
    examples = record.get("examples", [])
    if isinstance(examples, list):
        parts.extend([str(example) for example in examples])
    
    # Add tags
    tags = record.get("tags", [])
    if isinstance(tags, list):
        parts.extend([str(tag) for tag in tags])
    
    # Add business applications
    business_apps = record.get("business_applications", [])
    if isinstance(business_apps, list):
        parts.extend([str(app) for app in business_apps])
    
    # Add technology stack
    tech_stack = record.get("technology_stack", [])
    if isinstance(tech_stack, list):
        parts.extend([str(tech) for tech in tech_stack])
    
    return "\n".join([p for p in parts if p])

def build_innovation_doc(record: Dict[str, Any], vectors: Dict[str, List[float]]) -> Dict[str, Any]:
    """Build the Elasticsearch document structure for innovative ideas."""
    doc = {
        "idea_id": record.get("id"),
        "type": record.get("type"),
        "category": record.get("category"),
        "title": record.get("title"),
        "summary": record.get("summary"),
        "detailed_description": record.get("detailed_description"),
        "impact_level": record.get("impact_level"),
        "implementation_effort": record.get("implementation_effort"),
        "examples": record.get("examples", []),
        "tags": record.get("tags", []),
        "business_applications": record.get("business_applications", []),
        "technology_stack": record.get("technology_stack", [])
    }
    
    # Add all vector fields
    doc.update(vectors)
    return doc

def to_slug(value: str) -> str:
    """Convert string to URL-friendly slug."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") if value else ""

def innovation_id_builder(record: Dict[str, Any]) -> str:
    """Generate unique ID for innovative ideas."""
    idea_id = record.get("id")
    if idea_id:
        return str(idea_id)
    
    # Fallback to title-based slug
    title = record.get("title", "")
    return to_slug(title) or f"idea-{hash(str(record)) % 10000}"

def index_innovative_ideas(
    index_name: str,
    records: Iterable[Dict[str, Any]],
) -> Tuple[int, int]:
    """Index innovative ideas with multi-model embeddings."""
    actions: List[Dict[str, Any]] = []
    
    for record in records:
        text_for_embedding = concatenate_innovation_fields(record)
        if not text_for_embedding:
            print(f"No text found for record {record.get('id', 'unknown')}, skipping")
            continue
            
        doc_id = innovation_id_builder(record)
        
        # Check if we're updating existing document
        existing_doc = None
        if UPDATE_EXISTING:
            existing_doc = get_existing_document(index_name, doc_id)
        
        # Generate embeddings for active models
        vectors = {}
        for model_key in ACTIVE_MODELS:
            if model_key not in EMBEDDING_MODELS:
                print(f"Warning: Model {model_key} not found in EMBEDDING_MODELS")
                continue
                
            model_config = EMBEDDING_MODELS[model_key]
            vector_field = model_config["vector_field"]
            
            try:
                vector = get_embedding(text_for_embedding, model_key)
                vectors[vector_field] = vector
                print(f"Generated {model_key} embedding ({len(vector)} dims) for idea {doc_id}")
            except Exception as e:
                print(f"Failed to generate {model_key} embedding for idea {doc_id}: {e}")
                continue
        
        if not vectors:
            print(f"No vectors generated for idea {doc_id}, skipping")
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
            new_doc = build_innovation_doc(record, vectors)
            doc.update(new_doc)
        else:
            # Create fresh document
            doc = build_innovation_doc(record, vectors)
        
        actions.append({
            "_index": index_name,
            "_id": doc_id,
            "_source": doc
        })
    
    # Bulk index
    if actions:
        try:
            success, errors = bulk(es, actions, refresh=True)
            return success, len(errors)
        except Exception as e:
            print(f"Bulk indexing failed: {e}")
            return 0, len(actions)
    
    return 0, 0

def main():
    """Main execution function."""
    print("🚀 Starting Innovative Ideas Upload")
    print("=" * 50)
    print(f"Active models: {ACTIVE_MODELS}")
    print(f"Update existing documents: {UPDATE_EXISTING}")
    print(f"Recreate indices: {RECREATE_INDICES}")
    print(f"JSON file: {INNOVATIVE_IDEAS_JSON_PATH}")
    print()
    
    # Define base mapping for innovative ideas
    innovative_ideas_base_mapping = {
        "idea_id": {"type": "keyword"},
        "type": {"type": "keyword"},
        "category": {"type": "keyword"},
        "title": {"type": "text"},
        "summary": {"type": "text"},
        "detailed_description": {"type": "text"},
        "impact_level": {"type": "keyword"},
        "implementation_effort": {"type": "keyword"},
        "examples": {"type": "text"},
        "tags": {"type": "keyword"},
        "business_applications": {"type": "keyword"},
        "technology_stack": {"type": "keyword"}
    }
    
    # Create/update index with multi-vector support
    print(f"Setting up index: {INNOVATIVE_IDEAS_INDEX}")
    ensure_index_with_multi_vectors(INNOVATIVE_IDEAS_INDEX, innovative_ideas_base_mapping)
    
    # Load and process innovative ideas
    print(f"\nLoading data from: {INNOVATIVE_IDEAS_JSON_PATH}")
    try:
        ideas_records = read_json_list(INNOVATIVE_IDEAS_JSON_PATH)
        print(f"Found {len(ideas_records)} innovative ideas")
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return 1
    
    # Index the documents
    start_time = time.time()
    success_count, error_count = index_innovative_ideas(
        INNOVATIVE_IDEAS_INDEX,
        ideas_records
    )
    
    elapsed_time = time.time() - start_time
    
    # Results summary
    print("\n" + "=" * 50)
    print("📊 Upload Results Summary")
    print("=" * 50)
    print(f"✅ Successfully indexed: {success_count} documents")
    print(f"❌ Errors: {error_count} documents")
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"📍 Index: {INNOVATIVE_IDEAS_INDEX}")
    
    if success_count > 0:
        print(f"\n🎉 Success! Your innovative ideas are now searchable!")
        print(f"   Try queries like:")
        print(f"   - 'digital twins for manufacturing'")
        print(f"   - 'AI agents for customer service'")
        print(f"   - 'voice interfaces for healthcare'")
        
        # Test connectivity
        try:
            total_docs = es.count(index=INNOVATIVE_IDEAS_INDEX)["count"]
            print(f"\n📈 Total documents in {INNOVATIVE_IDEAS_INDEX}: {total_docs}")
        except Exception as e:
            print(f"\n⚠️  Could not verify document count: {e}")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

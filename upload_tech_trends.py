#!/usr/bin/env python3
"""
Upload script for tech trends innovation data.
This script uploads tech trends innovation data with OpenAI embeddings.
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
TECH_TRENDS_JSON_PATH = "tech_trends_innovation.json"
TECH_TRENDS_INDEX = "tech_trends_innovation"

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

def concatenate_tech_trend_fields(record: Dict[str, Any]) -> str:
    """Extract and concatenate text fields for embedding generation."""
    parts: List[str] = [
        record.get("title") or "",
        record.get("summary") or "",
        record.get("detailed_description") or "",
        record.get("category") or "",
        record.get("strategic_value") or "",
        record.get("maturity_level") or "",
        record.get("adoption_urgency") or "",
        record.get("competitive_impact") or "",
    ]
    
    # Add implementation considerations
    impl_considerations = record.get("implementation_considerations", {})
    if isinstance(impl_considerations, dict):
        # Add critical success factors
        critical_factors = impl_considerations.get("critical_success_factors", [])
        if isinstance(critical_factors, list):
            parts.extend([str(factor) for factor in critical_factors])
        
        # Add common pitfalls
        pitfalls = impl_considerations.get("common_pitfalls", [])
        if isinstance(pitfalls, list):
            parts.extend([str(pitfall) for pitfall in pitfalls])
    
    # Add real world applications
    applications = record.get("real_world_applications", [])
    if isinstance(applications, list):
        for app in applications:
            if isinstance(app, dict) and "example" in app:
                parts.append(str(app["example"]))
    
    return "\n".join([p for p in parts if p])

def build_tech_trend_doc(record: Dict[str, Any], vectors: Dict[str, List[float]]) -> Dict[str, Any]:
    """Build the Elasticsearch document structure for tech trends."""
    doc = {
        "trend_id": record.get("id"),
        "category": record.get("category"),
        "title": record.get("title"),
        "summary": record.get("summary"),
        "detailed_description": record.get("detailed_description"),
        "strategic_value": record.get("strategic_value"),
        "maturity_level": record.get("maturity_level"),
        "adoption_urgency": record.get("adoption_urgency"),
        "competitive_impact": record.get("competitive_impact"),
        "implementation_considerations": record.get("implementation_considerations", {}),
        "real_world_applications": record.get("real_world_applications", [])
    }
    
    # Add all vector fields
    doc.update(vectors)
    return doc

def to_slug(value: str) -> str:
    """Convert string to URL-friendly slug."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") if value else ""

def tech_trend_id_builder(record: Dict[str, Any]) -> str:
    """Generate unique ID for tech trends."""
    trend_id = record.get("id")
    if trend_id:
        return str(trend_id)
    
    # Fallback to title-based slug
    title = record.get("title", "")
    return to_slug(title) or f"trend-{hash(str(record)) % 10000}"

def index_tech_trends(
    index_name: str,
    records: Iterable[Dict[str, Any]],
) -> Tuple[int, int]:
    """Index tech trends with multi-model embeddings."""
    actions: List[Dict[str, Any]] = []
    
    for record in records:
        text_for_embedding = concatenate_tech_trend_fields(record)
        if not text_for_embedding:
            print(f"No text found for record {record.get('id', 'unknown')}, skipping")
            continue
            
        doc_id = tech_trend_id_builder(record)
        
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
                print(f"Generated {model_key} embedding ({len(vector)} dims) for trend {doc_id}")
            except Exception as e:
                print(f"Failed to generate {model_key} embedding for trend {doc_id}: {e}")
                continue
        
        if not vectors:
            print(f"No vectors generated for trend {doc_id}, skipping")
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
            new_doc = build_tech_trend_doc(record, vectors)
            doc.update(new_doc)
        else:
            # Create fresh document
            doc = build_tech_trend_doc(record, vectors)
        
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
    print("🚀 Starting Tech Trends Innovation Upload")
    print("=" * 50)
    print(f"Active models: {ACTIVE_MODELS}")
    print(f"Update existing documents: {UPDATE_EXISTING}")
    print(f"Recreate indices: {RECREATE_INDICES}")
    print(f"JSON file: {TECH_TRENDS_JSON_PATH}")
    print()
    
    # Define base mapping for tech trends
    tech_trends_base_mapping = {
        "trend_id": {"type": "keyword"},
        "category": {"type": "keyword"},
        "title": {"type": "text"},
        "summary": {"type": "text"},
        "detailed_description": {"type": "text"},
        "strategic_value": {"type": "text"},
        "maturity_level": {"type": "keyword"},
        "adoption_urgency": {"type": "keyword"},
        "competitive_impact": {"type": "keyword"},
        "implementation_considerations": {
            "type": "object",
            "properties": {
                "critical_success_factors": {"type": "text"},
                "common_pitfalls": {"type": "text"}
            }
        },
        "real_world_applications": {
            "type": "nested",
            "properties": {
                "example": {"type": "text"}
            }
        }
    }
    
    # Create/update index with multi-vector support
    print(f"Setting up index: {TECH_TRENDS_INDEX}")
    ensure_index_with_multi_vectors(TECH_TRENDS_INDEX, tech_trends_base_mapping)
    
    # Load and process tech trends
    print(f"\nLoading data from: {TECH_TRENDS_JSON_PATH}")
    try:
        trends_records = read_json_list(TECH_TRENDS_JSON_PATH)
        print(f"Found {len(trends_records)} tech trends")
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return 1
    
    # Index the documents
    start_time = time.time()
    success_count, error_count = index_tech_trends(
        TECH_TRENDS_INDEX,
        trends_records
    )
    
    elapsed_time = time.time() - start_time
    
    # Results summary
    print("\n" + "=" * 50)
    print("📊 Upload Results Summary")
    print("=" * 50)
    print(f"✅ Successfully indexed: {success_count} documents")
    print(f"❌ Errors: {error_count} documents")
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"📍 Index: {TECH_TRENDS_INDEX}")
    
    if success_count > 0:
        print(f"\n🎉 Success! Your tech trends are now searchable!")
        print(f"   Try queries like:")
        print(f"   - 'generative AI for content creation'")
        print(f"   - 'edge computing for real-time processing'")
        print(f"   - 'quantum algorithms for optimization'")
        print(f"   - 'decentralized identity systems'")
        
        # Test connectivity
        try:
            total_docs = es.count(index=TECH_TRENDS_INDEX)["count"]
            print(f"\n📈 Total documents in {TECH_TRENDS_INDEX}: {total_docs}")
        except Exception as e:
            print(f"\n⚠️  Could not verify document count: {e}")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

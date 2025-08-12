#!/usr/bin/env python3
"""
Check Actual Elasticsearch Index Names

This script connects to your Elasticsearch instance and lists all actual index names,
helping you identify what indices really exist vs what the app is looking for.
"""

import os
import sys
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import json

# Load environment variables
load_dotenv()

def get_first_env(names, default=None):
    """Get the first available environment variable"""
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default

def connect_to_elasticsearch():
    """Connect to Elasticsearch and return client"""
    print("🔌 Connecting to Elasticsearch...")
    print("=" * 50)
    
    # Get Elasticsearch configuration (same as main.py)
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
    
    CLOUD_ID = get_first_env(["CLOUD_ID", "ELASTIC_CLOUD_ID"])
    API_KEY = get_first_env(["ES_API_KEY", "ELASTIC_API_KEY", "API_KEY"])
    
    print(f"ES_URL: {ES_URL}")
    print(f"ES_USER: {ES_USER}")
    print(f"CLOUD_ID: {'Set' if CLOUD_ID else 'Not set'}")
    print(f"API_KEY: {'Set' if API_KEY else 'Not set'}")
    
    try:
        if CLOUD_ID and API_KEY:
            print("\n🌩️ Using Elastic Cloud connection...")
            es = Elasticsearch(
                cloud_id=CLOUD_ID,
                api_key=API_KEY,
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3
            )
        else:
            # Handle port in URL
            if ES_PORT and ":" not in ES_URL.split("//", 1)[-1].split("/", 1)[0]:
                ES_URL = f"{ES_URL}:{ES_PORT}"
                
            print(f"\n🔗 Using direct connection to {ES_URL}...")
            es = Elasticsearch(
                ES_URL,
                http_auth=(ES_USER, ES_PASS),
                verify_certs=False,
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3
            )
        
        # Test connection
        if es.ping():
            info = es.info()
            version = info.get('version', {}).get('number', 'unknown')
            cluster_name = info.get('cluster_name', 'unknown')
            print(f"✅ Connected to Elasticsearch {version}")
            print(f"✅ Cluster: {cluster_name}")
            return es
        else:
            print("❌ Failed to ping Elasticsearch")
            return None
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None

def list_all_indices(es):
    """List all indices with details"""
    print("\n📋 All Existing Indices...")
    print("=" * 70)
    
    try:
        # Get all indices
        indices_info = es.indices.get("*")
        
        if not indices_info:
            print("  No indices found")
            return []
        
        # Sort indices by name
        sorted_indices = sorted(indices_info.keys())
        
        print(f"{'Index Name':<35} {'Documents':<12} {'Size':<10}")
        print("-" * 70)
        
        index_names = []
        
        for index_name in sorted_indices:
            try:
                # Get document count
                count_result = es.count(index=index_name)
                doc_count = count_result.get('count', 0)
                
                # Get index stats for size
                stats = es.indices.stats(index=index_name)
                size_bytes = stats.get('indices', {}).get(index_name, {}).get('total', {}).get('store', {}).get('size_in_bytes', 0)
                size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
                
                print(f"{index_name:<35} {doc_count:<12} {size_mb:.1f} MB")
                index_names.append(index_name)
                
            except Exception as e:
                print(f"{index_name:<35} {'Error':<12} {str(e)[:20]}")
                index_names.append(index_name)
        
        return index_names
        
    except Exception as e:
        print(f"❌ Error listing indices: {e}")
        return []

def show_expected_vs_actual(es):
    """Show what the app expects vs what actually exists"""
    print("\n🔍 Expected vs Actual Index Names...")
    print("=" * 70)
    
    # What the app is currently configured to look for
    BASE_INDEX = os.getenv("INDEX", "providers")
    CLOUDS_INDEX = f"cloud_{BASE_INDEX}" if BASE_INDEX == "providers" else f"{BASE_INDEX}_providers"
    BEST_PRACTICES_INDEX = f"cloud_best_practices" if BASE_INDEX == "providers" else f"{BASE_INDEX}_best_practices"
    INNOVATIVE_IDEAS_INDEX = "innovative_ideas"
    TECH_TRENDS_INDEX = "tech_trends_innovation"
    
    expected_indices = {
        "Cloud Providers": CLOUDS_INDEX,
        "Best Practices": BEST_PRACTICES_INDEX,
        "Innovative Ideas": INNOVATIVE_IDEAS_INDEX,
        "Tech Trends": TECH_TRENDS_INDEX
    }
    
    print(f"BASE_INDEX environment variable: '{BASE_INDEX}'")
    print()
    print(f"{'Description':<20} {'Expected Name':<35} {'Status'}")
    print("-" * 70)
    
    for description, expected_name in expected_indices.items():
        try:
            if es.indices.exists(index=expected_name):
                count_result = es.count(index=expected_name)
                doc_count = count_result.get('count', 0)
                status = f"✅ EXISTS ({doc_count} docs)"
            else:
                status = "❌ NOT FOUND"
        except Exception as e:
            status = f"❌ ERROR: {str(e)[:20]}"
        
        print(f"{description:<20} {expected_name:<35} {status}")

def find_similar_indices(es, target_patterns):
    """Find indices that match common patterns"""
    print("\n🔎 Looking for Similar Index Names...")
    print("=" * 70)
    
    try:
        all_indices = es.indices.get("*")
        if not all_indices:
            print("  No indices to search")
            return
        
        patterns = [
            "cloud", "provider", "best_practice", "innovative", "tech_trend"
        ]
        
        print("Indices containing common keywords:")
        print()
        
        for pattern in patterns:
            matching = [idx for idx in all_indices.keys() if pattern in idx.lower()]
            if matching:
                print(f"  '{pattern}' pattern:")
                for idx in sorted(matching):
                    try:
                        count = es.count(index=idx).get('count', 0)
                        print(f"    - {idx} ({count} docs)")
                    except:
                        print(f"    - {idx} (count unknown)")
                print()
        
    except Exception as e:
        print(f"❌ Error searching for similar indices: {e}")

def generate_fix_suggestions(es):
    """Generate suggestions based on what we find"""
    print("\n💡 Fix Suggestions...")
    print("=" * 50)
    
    try:
        all_indices = list(es.indices.get("*").keys())
        
        # Look for likely cloud provider indices
        cloud_candidates = [idx for idx in all_indices if 'cloud' in idx.lower() and 'provider' in idx.lower()]
        practice_candidates = [idx for idx in all_indices if 'practice' in idx.lower() or 'best' in idx.lower()]
        
        print("Based on existing indices, consider these options:")
        print()
        
        if cloud_candidates:
            print("1. 🔧 Update INDEX environment variable:")
            for candidate in cloud_candidates:
                if candidate == "cloud_providers":
                    print(f"   export INDEX=providers  # Will create: cloud_providers ✅")
                elif candidate.startswith("cloud_"):
                    base = candidate.replace("cloud_", "").replace("_providers", "")
                    print(f"   export INDEX={base}  # Will create: {candidate}")
        
        if practice_candidates:
            print("\n2. 📋 Best practices indices found:")
            for candidate in practice_candidates:
                print(f"   - {candidate}")
        
        print(f"\n3. 🔍 Manual index name check:")
        print(f"   - Current BASE_INDEX: '{os.getenv('INDEX', 'providers')}'")
        print(f"   - This creates cloud index: 'cloud_{os.getenv('INDEX', 'providers')}'")
        print(f"   - And practices index: 'cloud_best_practices'")
        
    except Exception as e:
        print(f"❌ Error generating suggestions: {e}")

def main():
    """Main function to check actual index names"""
    print("🔍 Elasticsearch Index Name Checker")
    print("=" * 60)
    
    # Connect to Elasticsearch
    es = connect_to_elasticsearch()
    if not es:
        print("\n❌ Cannot proceed without Elasticsearch connection")
        return False
    
    # List all indices
    actual_indices = list_all_indices(es)
    
    # Show expected vs actual
    show_expected_vs_actual(es)
    
    # Find similar indices
    find_similar_indices(es, ["cloud", "provider", "practice"])
    
    # Generate fix suggestions
    generate_fix_suggestions(es)
    
    print(f"\n📊 Summary:")
    print(f"  Total indices found: {len(actual_indices)}")
    print(f"  Connection successful: ✅")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

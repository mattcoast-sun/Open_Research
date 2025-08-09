#!/usr/bin/env python3
"""
Debug script for Elasticsearch connection issues

This script helps diagnose and fix Elasticsearch connection problems,
particularly the version compatibility issues seen in the vector search.
"""

import os
import sys
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()


def get_first_env(names: list, default: str = None) -> str:
    """Get the first available environment variable."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def test_elasticsearch_connection():
    """Test Elasticsearch connection with different configurations."""
    
    print("Elasticsearch Connection Diagnostics")
    print("=" * 50)
    
    # Get configuration
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
    
    CLOUD_ID = get_first_env(["CLOUD_ID", "ELASTIC_CLOUD_ID"]) or None
    API_KEY = get_first_env(["ES_API_KEY", "ELASTIC_API_KEY", "API_KEY"]) or None
    
    print(f"Configuration:")
    print(f"  ES_URL: {ES_URL}")
    print(f"  ES_USER: {ES_USER}")
    print(f"  CLOUD_ID: {'***set***' if CLOUD_ID else 'Not set'}")
    print(f"  API_KEY: {'***set***' if API_KEY else 'Not set'}")
    print()
    
    # Test connection configurations
    configs_to_test = []
    
    if CLOUD_ID and API_KEY:
        configs_to_test.append({
            "name": "Cloud ID + API Key",
            "config": {
                "cloud_id": CLOUD_ID,
                "api_key": API_KEY,
                "request_timeout": 30
            }
        })
        
        configs_to_test.append({
            "name": "Cloud ID + API Key (with headers)",
            "config": {
                "cloud_id": CLOUD_ID,
                "api_key": API_KEY,
                "request_timeout": 30,
                "headers": {"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
            }
        })
    
    configs_to_test.append({
        "name": "URL + Auth",
        "config": {
            "hosts": [ES_URL],
            "http_auth": (ES_USER, ES_PASS),
            "verify_certs": False,
            "request_timeout": 30
        }
    })
    
    configs_to_test.append({
        "name": "URL + Auth (with headers)",
        "config": {
            "hosts": [ES_URL],
            "http_auth": (ES_USER, ES_PASS),
            "verify_certs": False,
            "request_timeout": 30,
            "headers": {"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
        }
    })
    
    # Test each configuration
    for i, test_config in enumerate(configs_to_test, 1):
        print(f"Test {i}: {test_config['name']}")
        print("-" * 30)
        
        try:
            es = Elasticsearch(**test_config['config'])
            
            # Test basic connection
            info = es.info()
            version = info.get('version', {}).get('number', 'unknown')
            cluster_name = info.get('cluster_name', 'unknown')
            
            print(f"✅ Connection successful!")
            print(f"   Elasticsearch version: {version}")
            print(f"   Cluster name: {cluster_name}")
            
            # Test index listing
            try:
                indices = list(es.indices.get_alias().keys())
                print(f"   Available indices: {len(indices)}")
                
                # Look for cloud-related indices
                cloud_indices = [idx for idx in indices if 'cloud' in idx.lower() or 'provider' in idx.lower()]
                if cloud_indices:
                    print(f"   Cloud-related indices: {cloud_indices}")
                else:
                    print(f"   No cloud-related indices found")
                    
            except Exception as e:
                print(f"   ⚠️ Could not list indices: {e}")
            
            # Test a simple search
            try:
                # Try to search for any documents
                search_result = es.search(
                    index="*",
                    query={"match_all": {}},
                    size=1,
                    timeout="10s"
                )
                doc_count = search_result.get('hits', {}).get('total', {})
                if isinstance(doc_count, dict):
                    doc_count = doc_count.get('value', 0)
                print(f"   Total documents across all indices: {doc_count}")
                
            except Exception as e:
                print(f"   ⚠️ Could not perform test search: {e}")
            
            print(f"   ✅ This configuration works!")
            print()
            
            # If we found a working configuration, stop testing
            break
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print()
            continue
    
    else:
        print("❌ All connection attempts failed!")
        print("\nTroubleshooting suggestions:")
        print("1. Check your environment variables are set correctly")
        print("2. Verify your Elasticsearch server is running")
        print("3. Check network connectivity")
        print("4. Verify credentials and permissions")
        print("5. Try updating the Elasticsearch client library:")
        print("   pip install 'elasticsearch>=8.0.0,<9.0.0'")


def test_vector_search_compatibility():
    """Test if vector search queries work with the current setup."""
    
    print("\nVector Search Compatibility Test")
    print("=" * 40)
    
    # This will test with the working configuration found above
    try:
        from main import es, CLOUDS_INDEX, BEST_PRACTICES_INDEX
        
        if es is None:
            print("❌ No Elasticsearch client available from main.py")
            return
        
        print(f"Testing with indices: {CLOUDS_INDEX}, {BEST_PRACTICES_INDEX}")
        
        # Test a simple vector search query structure
        test_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'body_vector') + 1.0",
                        "params": {"query_vector": [0.1] * 384}  # Dummy vector
                    }
                }
            },
            "size": 1
        }
        
        # Test on clouds index
        try:
            result = es.search(index=CLOUDS_INDEX, **test_query)
            print(f"✅ Vector search works on {CLOUDS_INDEX}")
            print(f"   Found {result['hits']['total']['value']} documents")
        except Exception as e:
            print(f"❌ Vector search failed on {CLOUDS_INDEX}: {e}")
        
        # Test on best practices index
        try:
            result = es.search(index=BEST_PRACTICES_INDEX, **test_query)
            print(f"✅ Vector search works on {BEST_PRACTICES_INDEX}")
            print(f"   Found {result['hits']['total']['value']} documents")
        except Exception as e:
            print(f"❌ Vector search failed on {BEST_PRACTICES_INDEX}: {e}")
            
    except ImportError as e:
        print(f"❌ Could not import from main.py: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    test_elasticsearch_connection()
    test_vector_search_compatibility()

#!/usr/bin/env python3
"""
Elasticsearch Indices Diagnostic Test

This script tests the Elasticsearch connection and checks for required indices.
It also provides commands to recreate missing indices if needed.
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

def test_elasticsearch_connection():
    """Test Elasticsearch connection and return client"""
    print("🔍 Testing Elasticsearch Connection...")
    print("=" * 50)
    
    # Get Elasticsearch configuration
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
            print(f"✅ Connected to Elasticsearch {version}")
            return es
        else:
            print("❌ Failed to ping Elasticsearch")
            return None
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None

def check_indices(es):
    """Check for required indices"""
    print("\n🔍 Checking Required Indices...")
    print("=" * 50)
    
    # Expected indices based on the error
    expected_indices = [
        "cloud_providers_providers",
        "cloud_providers_best_practices", 
        "innovative_ideas",
        "tech_trends_innovation"
    ]
    
    existing_indices = []
    missing_indices = []
    
    for index_name in expected_indices:
        try:
            if es.indices.exists(index=index_name):
                # Get document count
                count_result = es.count(index=index_name)
                doc_count = count_result.get('count', 0)
                print(f"✅ {index_name} - {doc_count} documents")
                existing_indices.append(index_name)
            else:
                print(f"❌ {index_name} - NOT FOUND")
                missing_indices.append(index_name)
        except Exception as e:
            print(f"❌ {index_name} - ERROR: {e}")
            missing_indices.append(index_name)
    
    return existing_indices, missing_indices

def list_all_indices(es):
    """List all existing indices"""
    print("\n📋 All Existing Indices...")
    print("=" * 50)
    
    try:
        indices = es.indices.get_alias("*")
        if indices:
            for index_name in sorted(indices.keys()):
                try:
                    count_result = es.count(index=index_name)
                    doc_count = count_result.get('count', 0)
                    print(f"  📁 {index_name} - {doc_count} documents")
                except:
                    print(f"  📁 {index_name} - (count unavailable)")
        else:
            print("  No indices found")
    except Exception as e:
        print(f"  Error listing indices: {e}")

def check_sample_documents(es, existing_indices):
    """Check sample documents from existing indices"""
    print("\n📄 Sample Documents...")
    print("=" * 50)
    
    for index_name in existing_indices[:2]:  # Check first 2 indices
        try:
            result = es.search(index=index_name, size=1)
            if result['hits']['hits']:
                doc = result['hits']['hits'][0]
                source_keys = list(doc['_source'].keys())
                print(f"\n{index_name} sample document:")
                print(f"  ID: {doc['_id']}")
                print(f"  Fields: {source_keys}")
                
                # Check for vector fields
                vector_fields = [k for k in source_keys if 'vector' in k.lower()]
                if vector_fields:
                    print(f"  Vector fields: {vector_fields}")
                else:
                    print(f"  ⚠️ No vector fields found!")
            else:
                print(f"\n{index_name}: No documents found")
        except Exception as e:
            print(f"\n{index_name} error: {e}")

def generate_fix_commands(missing_indices):
    """Generate commands to fix missing indices"""
    print("\n🛠️ How to Fix Missing Indices...")
    print("=" * 50)
    
    if not missing_indices:
        print("✅ No missing indices - everything looks good!")
        return
    
    print("Missing indices need to be created. Here are your options:")
    
    print("\n1. 📤 Upload data to create indices:")
    print("   python upload_data_new.py")
    print("   # or")
    print("   python upload_data_multi_model.py")
    
    print("\n2. 🔧 If you have existing data files:")
    print("   # Make sure these files exist:")
    for index in missing_indices:
        if "providers" in index:
            print("   - clouds.json")
        elif "best_practices" in index:
            print("   - cloud_best_practices.json")
        elif "innovative_ideas" in index:
            print("   - innovative_ideas.json")
        elif "tech_trends" in index:
            print("   - tech_trends_innovation.json")
    
    print("\n3. 🚀 For Railway deployment:")
    print("   # Set environment variables and run upload script")
    print("   export UPDATE_EXISTING=false")
    print("   export RECREATE_INDICES=true") 
    print("   python upload_data_new.py")

def main():
    """Main diagnostic function"""
    print("🏥 Elasticsearch Diagnostic Test")
    print("=" * 60)
    
    # Test connection
    es = test_elasticsearch_connection()
    if not es:
        print("\n❌ Cannot proceed without Elasticsearch connection")
        return False
    
    # Check indices
    existing_indices, missing_indices = check_indices(es)
    
    # List all indices
    list_all_indices(es)
    
    # Check sample documents
    if existing_indices:
        check_sample_documents(es, existing_indices)
    
    # Generate fix commands
    generate_fix_commands(missing_indices)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"✅ Existing indices: {len(existing_indices)}")
    print(f"❌ Missing indices: {len(missing_indices)}")
    
    if missing_indices:
        print(f"\n⚠️ Missing indices: {', '.join(missing_indices)}")
        print("Run the upload scripts to create missing indices.")
        return False
    else:
        print("\n🎉 All required indices are present!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3

import requests
import json

BASE_URL = "http://localhost:8000"

def inspect_vector_results():
    """Get and inspect the actual vector search results"""
    print("🔍 Inspecting vector search results structure...")
    
    payload = {
        "clarified_query": "Cloud providers for startups with cost-effective pricing",
        "sql_results": [],
        "max_results": 3  # Just get a few for inspection
    }
    
    response = requests.post(f"{BASE_URL}/vector-search", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Keys: {list(result.keys())}")
            print(f"Title: {result.get('title', 'NO TITLE')}")
            print(f"Document ID: {result.get('document_id', 'NO DOC ID')}")
            print(f"Similarity Score: {result.get('similarity_score', 'NO SCORE')} (type: {type(result.get('similarity_score'))})")
            print(f"Metadata: {result.get('metadata', 'NO METADATA')}")
            print(f"Content snippet: {str(result.get('content_snippet', 'NO CONTENT'))[:100]}...")
            
            # Check the structure
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'Not a dict'}")
        
        return results
    else:
        print(f"Error: {response.text}")
        return []

def test_with_real_data(vector_results):
    """Test the cohesive answer endpoint with real data"""
    print(f"\n📝 Testing cohesive answer with {len(vector_results)} real results...")
    
    payload = {
        "original_query": "Best cloud provider for startups",
        "vector_results": vector_results,
        "sql_context": []
    }
    
    print("Payload structure:")
    print(f"  original_query: {payload['original_query']}")
    print(f"  vector_results count: {len(payload['vector_results'])}")
    print(f"  sql_context: {payload['sql_context']}")
    
    response = requests.post(f"{BASE_URL}/cohesive-answer", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Answer length: {len(data.get('answer', ''))}")
        print(f"Sources count: {len(data.get('sources', []))}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        
        # Try to get more detailed error info
        try:
            error_detail = response.json()
            print(f"Error details: {json.dumps(error_detail, indent=2)}")
        except:
            print(f"Raw response: {response.content}")
        return False

if __name__ == "__main__":
    print("🔍 Debugging vector search data structure...\n")
    
    # First, inspect the actual data structure
    vector_results = inspect_vector_results()
    
    if vector_results:
        # Then test with that data
        test_with_real_data(vector_results)
    else:
        print("❌ Could not get vector results to test with")

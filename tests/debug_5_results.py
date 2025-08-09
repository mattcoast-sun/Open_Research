#!/usr/bin/env python3

import requests
import json

BASE_URL = "http://localhost:8000"

def get_5_vector_results():
    """Get exactly 5 vector search results like the failing test"""
    print("🔍 Getting exactly 5 vector search results...")
    
    payload = {
        "clarified_query": "Cloud providers for startups with cost-effective pricing",
        "sql_results": [],
        "max_results": 5  # Same as the failing test
    }
    
    response = requests.post(f"{BASE_URL}/vector-search", json=payload)
    print(f"Vector search status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        print(f"Got {len(results)} vector results")
        
        # Show each result structure
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Keys: {list(result.keys())}")
            print(f"Title: {result.get('title', 'NO TITLE')}")
            print(f"Document ID: {result.get('document_id', 'NO DOC ID')}")
            print(f"Similarity Score: {result.get('similarity_score', 'NO SCORE')} (type: {type(result.get('similarity_score'))})")
            
            metadata = result.get('metadata', {})
            print(f"Metadata type: {type(metadata)}")
            if isinstance(metadata, dict):
                print(f"Metadata keys: {list(metadata.keys())}")
                print(f"Metadata 'type': {metadata.get('type', 'NO TYPE')}")
            else:
                print(f"Metadata content: {metadata}")
        
        return results
    else:
        print(f"Vector search error: {response.text}")
        return []

def test_with_5_results(vector_results):
    """Test cohesive answer with exactly 5 results"""
    print(f"\n📝 Testing cohesive answer with 5 results...")
    
    payload = {
        "original_query": "Best cloud provider for startups",
        "vector_results": vector_results,
        "sql_context": []
    }
    
    print(f"Testing with {len(vector_results)} results")
    
    try:
        response = requests.post(f"{BASE_URL}/cohesive-answer", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Answer length: {len(data.get('answer', ''))}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print("Could not parse error as JSON")
                print(f"Raw content: {response.content}")
            return False
    except Exception as e:
        print(f"❌ Request exception: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing with exactly 5 results to match the failing test...\n")
    
    # Get exactly 5 results
    vector_results = get_5_vector_results()
    
    if vector_results:
        success = test_with_5_results(vector_results)
        if success:
            print("✅ 5-result test passed!")
        else:
            print("❌ 5-result test failed - this matches the main test issue!")
    else:
        print("❌ Could not get vector results")

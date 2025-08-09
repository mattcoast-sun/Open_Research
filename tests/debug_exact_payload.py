#!/usr/bin/env python3

import requests
import json

BASE_URL = "http://localhost:8000"

def get_vector_results_exact():
    """Get the exact vector search results that the test uses"""
    print("🔍 Getting vector search results exactly like the test...")
    
    payload = {
        "clarified_query": "Cloud providers for startups with cost-effective pricing",
        "sql_results": [],
        "max_results": 10  # Default value from test
    }
    
    response = requests.post(f"{BASE_URL}/vector-search", json=payload)
    print(f"Vector search status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        print(f"Got {len(results)} vector results")
        
        # Print a summary of each result
        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result.get('title', 'NO TITLE')[:50]}... Score: {result.get('similarity_score')}")
        
        return results
    else:
        print(f"Vector search error: {response.text}")
        return []

def test_cohesive_answer_exact(vector_results):
    """Test with the exact same structure as the main test"""
    print(f"\n📝 Testing cohesive answer with exact test payload...")
    
    payload = {
        "original_query": "Best cloud provider for startups",
        "vector_results": vector_results,
        "sql_context": []
    }
    
    print(f"Payload keys: {list(payload.keys())}")
    print(f"Original query: {payload['original_query']}")
    print(f"Vector results count: {len(payload['vector_results'])}")
    print(f"SQL context: {payload['sql_context']}")
    
    # Show structure of first result
    if vector_results:
        first_result = vector_results[0]
        print(f"First result keys: {list(first_result.keys())}")
        print(f"First result metadata type: {type(first_result.get('metadata'))}")
        if 'metadata' in first_result:
            metadata = first_result['metadata']
            if isinstance(metadata, dict):
                print(f"First result metadata keys: {list(metadata.keys())}")
    
    try:
        response = requests.post(f"{BASE_URL}/cohesive-answer", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Answer length: {len(data.get('answer', ''))}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            # Try to parse JSON error details
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print("Could not parse error as JSON")
            return False
    except Exception as e:
        print(f"❌ Request exception: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing with exact same structure as main test...\n")
    
    # Get the exact same vector results
    vector_results = get_vector_results_exact()
    
    if vector_results:
        # Test with those exact results
        success = test_cohesive_answer_exact(vector_results)
        if success:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
    else:
        print("❌ Could not get vector results")

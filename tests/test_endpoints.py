#!/usr/bin/env python3
"""
Quick test script for the FastAPI endpoints
Make sure to activate the virtual environment first:
source venv/bin/activate
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_clarification_agent():
    """Test clarification agent"""
    print("\n🤔 Testing clarification agent...")
    
    payload = {
        "user_query": "I need help choosing a cloud provider for my startup",
        "previous_questions": [],
        "previous_answers": []
    }
    
    response = requests.post(f"{BASE_URL}/clarification-agent", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Session ID: {data.get('session_id')}")
        print(f"Question: {data.get('question')}")
        print(f"Status: {data.get('status')}")
        return data.get('session_id'), data.get('question')
    else:
        print(f"Error: {response.text}")
        return None, None

def test_sql_query_generator():
    """Test SQL query generator"""
    print("\n📊 Testing SQL query generator...")
    
    payload = {
        "clarified_query": "Find cloud providers suitable for startups with cost-effective pricing and good support for web applications"
    }
    
    response = requests.post(f"{BASE_URL}/sql-query-generator", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Generated Query (first 200 chars): {data.get('sql_query', '')[:200]}...")
        print(f"Explanation: {data.get('explanation', '')[:150]}...")
        return data.get('sql_query')
    else:
        print(f"Error: {response.text}")
        return None

def test_vector_search():
    """Test vector search"""
    print("\n🔍 Testing vector search...")
    
    payload = {
        "clarified_query": "Cloud providers for startups with cost-effective pricing",
        "sql_results": [],
        "max_results": 5
    }
    
    response = requests.post(f"{BASE_URL}/vector-search", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Results found: {data.get('total_found')}")
        print(f"Search metadata: {data.get('search_metadata', {})}")
        
        results = data.get('results', [])
        for i, result in enumerate(results[:2]):  # Show first 2
            print(f"  Result {i+1}: {result.get('title')} (Score: {result.get('similarity_score')})")
        
        return results
    else:
        print(f"Error: {response.text}")
        return []

def test_cohesive_answer(vector_results=None):
    """Test cohesive answer generation"""
    print("\n📝 Testing cohesive answer generator...")
    
    # Add a small delay to prevent potential race conditions
    time.sleep(0.5)
    
    # Use real vector results if provided, otherwise use mock data
    if not vector_results:
        vector_results = [
            {
                "document_id": "test_doc_1",
                "title": "AWS for Startups",
                "content_snippet": "Amazon Web Services offers comprehensive cloud solutions...",
                "similarity_score": 0.85,
                "metadata": {"type": "cloud_provider", "provider_name": "AWS"}
            }
        ]
    
    payload = {
        "original_query": "Best cloud provider for startups",
        "vector_results": vector_results,
        "sql_context": []
    }
    
    # Debug info
    print(f"Sending payload with {len(vector_results)} vector results")
    if vector_results:
        print(f"First result keys: {list(vector_results[0].keys())}")
    
    response = requests.post(f"{BASE_URL}/cohesive-answer", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Answer length: {len(data.get('answer', ''))}")
        print(f"Confidence score: {data.get('confidence_score')}")
        print(f"Sources: {len(data.get('sources', []))}")
        print(f"Recommendations: {len(data.get('additional_recommendations', []))}")
        return True
    else:
        print(f"Error: {response.text}")
        # Try to get detailed error info
        try:
            error_detail = response.json()
            print(f"Error details: {json.dumps(error_detail, indent=2)}")
        except:
            print(f"Raw response content: {response.content}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting FastAPI endpoint tests...\n")
    
    # Test health first
    if not test_health():
        print("❌ Health check failed - is the server running?")
        return
    
    # Test each endpoint
    session_id, question = test_clarification_agent()
    sql_query = test_sql_query_generator()
    vector_results = test_vector_search()
    answer_success = test_cohesive_answer(vector_results)
    
    print("\n✅ All tests completed!")
    print(f"Clarification: {'✓' if session_id else '❌'}")
    print(f"SQL Generation: {'✓' if sql_query else '❌'}")
    print(f"Vector Search: {'✓' if vector_results else '❌'}")
    print(f"Answer Synthesis: {'✓' if answer_success else '❌'}")

if __name__ == "__main__":
    main()


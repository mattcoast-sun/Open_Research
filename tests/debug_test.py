#!/usr/bin/env python3

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint first"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def test_simple_answer():
    """Test the simple test-answer endpoint (no OpenAI)"""
    print("\n📝 Testing simple answer generator (no OpenAI)...")
    
    mock_results = [
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
        "vector_results": mock_results,
        "sql_context": []
    }
    
    try:
        response = requests.post(f"{BASE_URL}/test-answer", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Answer: {data.get('answer', '')[:100]}...")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def test_openai_answer():
    """Test the full cohesive-answer endpoint (with OpenAI)"""
    print("\n📝 Testing full answer generator (with OpenAI)...")
    
    mock_results = [
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
        "vector_results": mock_results,
        "sql_context": []
    }
    
    try:
        response = requests.post(f"{BASE_URL}/cohesive-answer", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Answer: {data.get('answer', '')[:100]}...")
            return True
        else:
            print(f"❌ Error: {response.text}")
            print(f"❌ Response content: {response.content}")
            return False
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting debug tests...\n")
    
    # Test each endpoint step by step
    health_ok = test_health()
    if not health_ok:
        print("❌ Health check failed - server might not be running")
        exit(1)
    
    simple_ok = test_simple_answer()
    if not simple_ok:
        print("❌ Simple answer test failed - check basic endpoint logic")
        exit(1)
    
    openai_ok = test_openai_answer()
    if not openai_ok:
        print("❌ OpenAI answer test failed - check OpenAI configuration")
    else:
        print("✅ All tests passed!")

#!/usr/bin/env python3
"""
Test script for the combined research pipeline endpoint
This demonstrates how to use the new /research-pipeline endpoint
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8001"

def test_combined_research_pipeline(question: str, **kwargs) -> Dict[str, Any]:
    """
    Test the combined research pipeline endpoint
    
    Args:
        question: The research question to ask
        **kwargs: Additional parameters like skip_clarification, max_results, etc.
    
    Returns:
        Dict containing the response from the API
    """
    
    # Prepare the request payload
    payload = {"question": question}
    payload.update(kwargs)
    
    print(f"🔍 Asking: {question}")
    print(f"📝 Request parameters: {json.dumps({k:v for k,v in payload.items() if k != 'question'}, indent=2)}")
    
    try:
        # Make the API call
        response = requests.post(f"{BASE_URL}/research-pipeline", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ Success! Processing took {data['processing_metadata']['total_processing_time']}")
            print(f"🎯 Confidence Score: {data['confidence_score']:.2f}")
            print(f"📚 Sources Found: {len(data['sources'])}")
            
            print(f"\n📋 **Enhanced Query:**")
            print(f"   {data['clarified_query']}")
            
            print(f"\n💡 **Answer:**")
            print(f"   {data['final_answer'][:300]}...")
            
            print(f"\n🔗 **Top Sources:**")
            for i, source in enumerate(data['sources'][:3], 1):
                print(f"   {i}. {source['title']} (Score: {source['similarity_score']})")
            
            print(f"\n📌 **Recommendations:**")
            for i, rec in enumerate(data['additional_recommendations'], 1):
                print(f"   {i}. {rec}")
            
            return data
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return {}
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - the research pipeline may be taking longer than expected")
        return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def main():
    """Run example tests"""
    print("🚀 Testing Combined Research Pipeline Endpoint\n")
    
    # Test 1: Basic question
    print("=" * 60)
    print("TEST 1: Basic Cloud Provider Question")
    print("=" * 60)
    test_combined_research_pipeline(
        "What is the best cloud provider for a startup?"
    )
    
    # Test 2: More specific question with options
    print("\n" + "=" * 60)
    print("TEST 2: Detailed Question with Custom Parameters")
    print("=" * 60)
    test_combined_research_pipeline(
        "How do I implement microservices architecture in the cloud?",
        max_results=15,
        skip_clarification=False
    )
    
    # Test 3: Skip clarification for direct processing
    print("\n" + "=" * 60)
    print("TEST 3: Skip Clarification Mode")
    print("=" * 60)
    test_combined_research_pipeline(
        "Best practices for cloud security",
        skip_clarification=True,
        max_results=5
    )

if __name__ == "__main__":
    main()

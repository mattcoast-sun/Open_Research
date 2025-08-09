#!/usr/bin/env python3
"""
Test script for the Cloud Ratings SQL endpoint

This script tests the new /cloud-ratings-sql endpoint with various
types of questions to demonstrate its functionality.
"""

import requests
import json
import sys
import time
from typing import Dict, Any


def test_endpoint(base_url: str, question: str, query_type: str = "analysis") -> Dict[str, Any]:
    """Test the cloud ratings SQL endpoint with a specific question."""
    
    endpoint = f"{base_url}/cloud-ratings-sql"
    
    payload = {
        "question": question,
        "query_type": query_type
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"TESTING: {question}")
        print(f"Query Type: {query_type}")
        print(f"{'='*60}")
        
        response = requests.post(endpoint, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ SUCCESS")
            print(f"Detected Query Type: {result.get('query_type', 'unknown')}")
            print(f"Confidence Score: {result.get('confidence_score', 0):.2f}")
            print(f"\n📊 Generated SQL Query:")
            print("-" * 40)
            print(result.get('sql_query', 'No query generated'))
            
            print(f"\n💡 Explanation:")
            print("-" * 40)
            print(result.get('explanation', 'No explanation provided'))
            
            print(f"\n🔍 Estimated Results:")
            print("-" * 40)
            print(result.get('estimated_results', 'No description provided'))
            
            print(f"\n💻 Example Usage:")
            print("-" * 40)
            print(result.get('example_usage', 'No example provided'))
            
            return result
            
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            return {"error": f"HTTP {response.status_code}", "details": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return {"error": "Connection failed", "details": str(e)}
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return {"error": "Unexpected error", "details": str(e)}


def main():
    """Main test function."""
    
    # Configuration
    base_url = "http://localhost:8000"  # Adjust if needed
    
    # Test questions covering different query types
    test_cases = [
        {
            "question": "What are the top 5 cloud providers by aggregate score?",
            "query_type": "ranking",
            "description": "Ranking query - should generate ORDER BY with LIMIT"
        },
        {
            "question": "Compare AWS, GCP, and Azure on AI capabilities",
            "query_type": "comparison", 
            "description": "Comparison query - should filter specific providers"
        },
        {
            "question": "Which cloud providers have cost efficiency above 8.0?",
            "query_type": "analysis",
            "description": "Analysis query - should use WHERE clause with threshold"
        },
        {
            "question": "Show me detailed information about IBM Cloud",
            "query_type": "specific",
            "description": "Specific provider query - should filter by provider name"
        },
        {
            "question": "What's the average sustainability score across all providers?",
            "query_type": "analysis",
            "description": "Statistical analysis - should use AVG() function"
        },
        {
            "question": "Which provider has the best performance score?",
            "query_type": "ranking",
            "description": "Best performer - should use MAX() or ORDER BY + LIMIT 1"
        },
        {
            "question": "List providers with both high AI capabilities and sustainability",
            "query_type": "analysis",
            "description": "Multi-criteria analysis - should use multiple WHERE conditions"
        }
    ]
    
    print("Cloud Ratings SQL Endpoint Test Suite")
    print("=" * 60)
    print(f"Testing endpoint: {base_url}/cloud-ratings-sql")
    print(f"Total test cases: {len(test_cases)}")
    
    # Check if server is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Server health check failed: {health_response.status_code}")
            print("Make sure the FastAPI server is running with: python main.py")
            sys.exit(1)
        else:
            print(f"✅ Server is running")
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to server at {base_url}")
        print("Make sure the FastAPI server is running with: python main.py")
        sys.exit(1)
    
    # Run test cases
    results = []
    successful_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}/{len(test_cases)}: {test_case['description']}")
        
        result = test_endpoint(
            base_url=base_url,
            question=test_case["question"],
            query_type=test_case["query_type"]
        )
        
        results.append({
            "test_case": test_case,
            "result": result
        })
        
        if "error" not in result:
            successful_tests += 1
        
        time.sleep(1)  # Brief pause between requests
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {len(test_cases) - successful_tests}")
    print(f"Success Rate: {(successful_tests / len(test_cases)) * 100:.1f}%")
    
    if successful_tests == len(test_cases):
        print(f"\n🎉 All tests passed! The endpoint is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Check the errors above.")
    
    # Show sample queries for manual testing
    print(f"\n📋 SAMPLE QUERIES FOR MANUAL TESTING:")
    print("-" * 40)
    for i, test_case in enumerate(test_cases[:3], 1):
        print(f"{i}. {test_case['question']}")
    
    print(f"\n💡 You can also test the endpoint using:")
    print(f"   - FastAPI docs: {base_url}/docs")
    print(f"   - cURL commands")
    print(f"   - Postman or similar tools")


if __name__ == "__main__":
    main()

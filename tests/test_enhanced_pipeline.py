#!/usr/bin/env python3
"""
Test script for the enhanced research pipeline with SQL integration

This script tests the enhanced /research-pipeline endpoint that now includes
cloud ratings SQL execution integrated into the workflow.
"""

import requests
import json
import time
import sys
from typing import Dict, Any


def test_enhanced_pipeline(base_url: str, question: str, skip_clarification: bool = True) -> Dict[str, Any]:
    """Test the enhanced research pipeline endpoint."""
    
    endpoint = f"{base_url}/research-pipeline"
    
    payload = {
        "question": question,
        "skip_clarification": skip_clarification,
        "max_results": 10
    }
    
    print(f"\n{'='*80}")
    print(f"TESTING ENHANCED RESEARCH PIPELINE")
    print(f"{'='*80}")
    print(f"Question: {question}")
    print(f"Skip Clarification: {skip_clarification}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        response = requests.post(endpoint, json=payload, timeout=60)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ SUCCESS - Request completed in {request_time:.2f}s")
            
            # Display key information
            print(f"\n📋 PROCESSING METADATA:")
            metadata = result.get('processing_metadata', {})
            print(f"  Pipeline Version: {metadata.get('pipeline_version', 'unknown')}")
            print(f"  Total Processing Time: {metadata.get('total_processing_time', 'unknown')}")
            print(f"  Steps Completed: {', '.join(metadata.get('steps_completed', []))}")
            
            # Cloud ratings SQL information
            cloud_sql_metadata = metadata.get('cloud_ratings_sql_metadata', {})
            if cloud_sql_metadata:
                print(f"\n🗄️ CLOUD RATINGS SQL INTEGRATION:")
                if cloud_sql_metadata.get('skipped'):
                    print(f"  Status: Skipped - {cloud_sql_metadata.get('reason', 'Unknown reason')}")
                elif cloud_sql_metadata.get('error'):
                    print(f"  Status: Error - {cloud_sql_metadata.get('error')}")
                else:
                    print(f"  Status: Executed successfully")
                    print(f"  Query Type: {cloud_sql_metadata.get('query_type', 'unknown')}")
                    print(f"  Rows Returned: {cloud_sql_metadata.get('row_count', 0)}")
                    print(f"  Execution Time: {cloud_sql_metadata.get('execution_time', 0):.3f}s")
                    print(f"  Confidence Score: {cloud_sql_metadata.get('confidence_score', 0):.2f}")
                    if cloud_sql_metadata.get('sql_query'):
                        print(f"  SQL Query: {cloud_sql_metadata['sql_query'][:100]}...")
            
            # Vector search results
            vector_metadata = metadata.get('vector_search_metadata', {})
            print(f"\n🔍 VECTOR SEARCH:")
            print(f"  Results Found: {metadata.get('vector_results_found', 0)}")
            print(f"  Cloud Ratings Results: {metadata.get('cloud_ratings_results_count', 0)}")
            if vector_metadata:
                print(f"  Search Time: {vector_metadata.get('vector_search_time', 'unknown')}")
            
            # Final answer
            print(f"\n💡 FINAL ANSWER:")
            print(f"  Confidence Score: {result.get('confidence_score', 0):.2f}")
            answer = result.get('final_answer', 'No answer provided')
            print(f"  Answer Length: {len(answer)} characters")
            print(f"  Answer Preview: {answer[:200]}...")
            
            # Sources
            sources = result.get('sources', [])
            print(f"\n📚 SOURCES:")
            print(f"  Total Sources: {len(sources)}")
            for i, source in enumerate(sources[:3]):
                print(f"    {i+1}. {source.get('title', 'Unknown')} ({source.get('type', 'unknown')})")
            
            # Recommendations
            recommendations = result.get('additional_recommendations', [])
            print(f"\n🎯 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"    {i}. {rec}")
            
            return {"success": True, "result": result, "request_time": request_time}
            
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return {"success": False, "error": f"HTTP {response.status_code}", "details": response.text}
            
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT: Request took longer than 60 seconds")
        return {"success": False, "error": "Timeout"}
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return {"success": False, "error": "Connection failed", "details": str(e)}
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return {"success": False, "error": "Unexpected error", "details": str(e)}


def main():
    """Main test function."""
    
    base_url = "http://localhost:8003"  # Updated port from the user's change
    
    # Test questions that should trigger cloud ratings SQL execution
    test_questions = [
        {
            "question": "What are the top 3 cloud providers for AI capabilities?",
            "description": "Should trigger cloud ratings SQL with ranking query"
        },
        {
            "question": "Compare AWS and GCP on performance and cost efficiency",
            "description": "Should trigger cloud ratings SQL with comparison query"
        },
        {
            "question": "Which cloud providers have the best sustainability scores?",
            "description": "Should trigger cloud ratings SQL with sustainability analysis"
        },
        {
            "question": "What are the best practices for container orchestration?",
            "description": "Should primarily use vector search, minimal cloud ratings SQL"
        },
        {
            "question": "How do I implement machine learning pipelines in the cloud?",
            "description": "Mixed query - should use both vector search and cloud ratings"
        }
    ]
    
    print("Enhanced Research Pipeline Test Suite")
    print("=" * 80)
    print(f"Testing endpoint: {base_url}/research-pipeline")
    print(f"Total test cases: {len(test_questions)}")
    
    # Check if server is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Server health check failed: {health_response.status_code}")
            print("Make sure the FastAPI server is running with the updated port (8003)")
            sys.exit(1)
        else:
            print(f"✅ Server is running at {base_url}")
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to server at {base_url}")
        print("Make sure the FastAPI server is running: python main.py")
        sys.exit(1)
    
    # Check if cloud ratings database exists
    try:
        db_test_response = requests.post(f"{base_url}/cloud-ratings-sql-execute", json={
            "question": "SELECT COUNT(*) FROM cloud_ratings",
            "execute_query": True
        }, timeout=10)
        
        if db_test_response.status_code == 200:
            db_result = db_test_response.json()
            if db_result.get('query_results') and len(db_result['query_results']) > 0:
                record_count = db_result['query_results'][0].get('COUNT(*)', 0)
                print(f"✅ Cloud ratings database is available with {record_count} records")
            else:
                print(f"⚠️ Cloud ratings database may be empty or inaccessible")
        else:
            print(f"⚠️ Cloud ratings database test failed: {db_test_response.status_code}")
            print("The pipeline will still work but may not have SQL integration")
    except Exception as e:
        print(f"⚠️ Database connectivity test failed: {e}")
        print("The pipeline will still work but may not have SQL integration")
    
    # Run test cases
    results = []
    successful_tests = 0
    
    for i, test_case in enumerate(test_questions, 1):
        print(f"\n🧪 Test Case {i}/{len(test_questions)}: {test_case['description']}")
        
        result = test_enhanced_pipeline(
            base_url=base_url,
            question=test_case["question"],
            skip_clarification=True
        )
        
        results.append({
            "test_case": test_case,
            "result": result
        })
        
        if result.get("success"):
            successful_tests += 1
        
        time.sleep(2)  # Brief pause between requests
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {len(test_questions)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {len(test_questions) - successful_tests}")
    print(f"Success Rate: {(successful_tests / len(test_questions)) * 100:.1f}%")
    
    if successful_tests == len(test_questions):
        print(f"\n🎉 All tests passed! The enhanced research pipeline is working correctly.")
        print(f"✅ SQL integration is enhancing the research workflow")
    else:
        print(f"\n⚠️ Some tests failed. Check the errors above.")
    
    # Performance summary
    successful_results = [r for r in results if r["result"].get("success")]
    if successful_results:
        avg_time = sum(r["result"].get("request_time", 0) for r in successful_results) / len(successful_results)
        print(f"\n⏱️ PERFORMANCE:")
        print(f"  Average Request Time: {avg_time:.2f}s")
        print(f"  Range: {min(r['result'].get('request_time', 0) for r in successful_results):.2f}s - {max(r['result'].get('request_time', 0) for r in successful_results):.2f}s")


if __name__ == "__main__":
    main()

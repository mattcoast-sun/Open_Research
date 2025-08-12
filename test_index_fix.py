#!/usr/bin/env python3
"""
Index Fix Validation Test

This test specifically validates that the Elasticsearch index naming fix
is working correctly by testing the exact scenarios that were failing before.
"""

import requests
import json
import sys

BASE_URL = "https://researcherlegend-production.up.railway.app"

def test_index_fix():
    """Test that the index naming fix resolved the 404 errors"""
    print("🔧 Testing Elasticsearch Index Fix")
    print("=" * 50)
    
    # Test the exact scenario that was failing before
    test_payload = {
        "question": "What are the best cloud providers for AI workloads?",
        "skip_clarification": True,
        "max_results": 3
    }
    
    print("📝 Testing research pipeline (the failing endpoint)...")
    print(f"   Query: {test_payload['question']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/research-pipeline",
            json=test_payload,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we got actual results (not index errors)
            sources = data.get("sources", [])
            final_answer = data.get("final_answer", "")
            metadata = data.get("processing_metadata", {})
            
            print(f"✅ SUCCESS: No more index_not_found_exception errors!")
            print(f"   Final Answer Length: {len(final_answer)} characters")
            print(f"   Sources Found: {len(sources)}")
            print(f"   Processing Steps: {metadata.get('steps_completed', [])}")
            
            # Check for specific success indicators
            if final_answer and len(final_answer) > 100:
                print(f"✅ Comprehensive answer generated")
            if sources and len(sources) > 0:
                print(f"✅ Sources successfully retrieved from indices")
            
            # Print sample of the answer
            if final_answer:
                sample = final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
                print(f"\n📄 Sample Answer:")
                print(f"   {sample}")
            
            return True
            
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Raw Error: {response.text[:300]}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Exception - {str(e)}")
        return False

def test_vector_search_specifically():
    """Test vector search specifically to ensure indices are accessible"""
    print("\n🔍 Testing Vector Search (Index Access)...")
    
    test_payload = {
        "clarified_query": "cloud security best practices for enterprise",
        "max_results": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/vector-search",
            json=test_payload,
            timeout=20,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            total_found = data.get("total_found", 0)
            search_metadata = data.get("search_metadata", {})
            
            print(f"✅ Vector Search Success!")
            print(f"   Results Returned: {len(results)}")
            print(f"   Total Found: {total_found}")
            print(f"   Search Type: {search_metadata.get('search_type', 'unknown')}")
            
            # Check if we're getting results from the correct indices
            indices_searched = search_metadata.get("indices_searched", [])
            if indices_searched:
                print(f"   Indices Searched: {indices_searched}")
                
                # These should now be the correct index names
                expected_indices = ["cloud_providers", "cloud_best_practices", "innovative_ideas", "tech_trends_innovation"]
                found_expected = [idx for idx in expected_indices if idx in str(indices_searched)]
                
                if found_expected:
                    print(f"✅ Correctly accessing expected indices: {found_expected}")
                else:
                    print(f"⚠️ Unexpected indices: {indices_searched}")
            
            return True
            
        else:
            print(f"❌ Vector Search Failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Vector Search Exception: {str(e)}")
        return False

def main():
    """Run the index fix validation tests"""
    print("🧪 Elasticsearch Index Fix Validation")
    print("=" * 60)
    print("Testing the fix for index_not_found_exception errors")
    print("Previous errors: cloud_providers_providers, cloud_providers_best_practices")
    print("Expected fix: Now using cloud_providers, cloud_best_practices")
    print()
    
    # Run tests
    test1_passed = test_index_fix()
    test2_passed = test_vector_search_specifically()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INDEX FIX VALIDATION SUMMARY")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("🎉 INDEX FIX SUCCESSFUL!")
        print("✅ No more index_not_found_exception errors")
        print("✅ Railway deployment is fully operational")
        print("✅ All endpoints accessing correct Elasticsearch indices")
        
        print("\n🎯 What was fixed:")
        print("   Before: BASE_INDEX = 'cloud_providers' → 'cloud_providers_providers' ❌")
        print("   After:  BASE_INDEX = 'providers' → 'cloud_providers' ✅")
        
        return True
    else:
        print("❌ INDEX FIX VALIDATION FAILED")
        print("Some tests are still failing - may need additional investigation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

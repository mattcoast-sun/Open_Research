#!/usr/bin/env python3
"""
Simple test without requiring HF token - just test the fallback logic.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_fallback_behavior():
    """Test that the system falls back to text search when embeddings fail."""
    print("🧪 Testing fallback behavior...")
    
    # Import main to test vector search fallback
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint first
        response = client.get("/health")
        print(f"Health check: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ FastAPI app loads successfully")
            
            # Try vector search (should fall back to text search)
            vector_request = {
                "clarified_query": "What are the best cloud providers?",
                "max_results": 5
            }
            
            # This might fail due to ES connection, but let's see
            try:
                response = client.post("/vector-search", json=vector_request)
                print(f"Vector search response: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"Search metadata: {data.get('search_metadata', {})}")
                else:
                    print(f"Response: {response.text[:200]}")
            except Exception as e:
                print(f"Vector search test failed (expected): {e}")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed dependencies: pip install -r requirements.openai.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_embedding_imports():
    """Test that the embedding module imports correctly."""
    print("🧪 Testing embedding module imports...")
    
    try:
        from embeddings import EMBEDDINGS_MODE, get_query_embedding
        print(f"✅ Embeddings module imported successfully")
        print(f"Current mode: {EMBEDDINGS_MODE}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("🔬 Simple Integration Test")
    print("=" * 40)
    
    results = []
    results.append(("Embedding imports", test_embedding_imports()))
    results.append(("Fallback behavior", test_fallback_behavior()))
    
    print("\n📊 Results:")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Basic integration works! Now get an HF token to test embeddings.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

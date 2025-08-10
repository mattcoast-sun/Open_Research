#!/usr/bin/env python3
"""
Direct test of OpenAI embeddings to verify the torch-free setup works.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_openai_direct():
    """Test OpenAI embeddings directly."""
    print("🧪 Testing OpenAI embeddings directly...")
    
    # Check if we have the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        test_query = "What are the best cloud providers for AI workloads?"
        
        print(f"Query: {test_query}")
        
        # Test with different OpenAI embedding models
        models_to_test = [
            "text-embedding-3-small",
            "text-embedding-ada-002"
        ]
        
        for model in models_to_test:
            try:
                print(f"\nTesting model: {model}")
                resp = client.embeddings.create(model=model, input=test_query)
                embedding = resp.data[0].embedding
                
                print(f"✅ Success! Model: {model}")
                print(f"   Dimensions: {len(embedding)}")
                print(f"   First 5 values: {embedding[:5]}")
                
                # Check if it matches your 384-dim index
                if len(embedding) == 384:
                    print(f"   ✅ Perfect match for your index!")
                else:
                    print(f"   ⚠️  Dimension mismatch with 384-dim index")
                
            except Exception as e:
                print(f"   ❌ Error with {model}: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_app_integration():
    """Test that the app works with OpenAI embeddings."""
    print("\n🧪 Testing app integration with OpenAI...")
    
    # Set environment for this test
    os.environ["EMBEDDINGS_MODE"] = "openai"
    os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-small"
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test vector search endpoint
        response = client.post("/vector-search", json={
            "clarified_query": "What are the best cloud providers for AI?",
            "max_results": 3
        })
        
        print(f"Vector search status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            search_metadata = data.get("search_metadata", {})
            search_type = search_metadata.get("search_type", "unknown")
            
            print(f"Search type: {search_type}")
            print(f"Results found: {data.get('total_found', 0)}")
            
            if search_type == "text_fallback":
                print("✅ Graceful fallback to text search working!")
            else:
                print("✅ Vector search working!")
            
            return True
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

def main():
    print("🔬 Direct OpenAI Embedding Test")
    print("=" * 40)
    
    results = []
    results.append(("OpenAI Direct", test_openai_direct()))
    results.append(("App Integration", test_app_integration()))
    
    print("\n📊 Results:")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Ready for Code Engine with OpenAI embeddings!")
        print("\nNext steps:")
        print("1. Build with: docker build --build-arg REQUIREMENTS_FILE=requirements.openai.txt -t myapp .")
        print("2. Deploy to Code Engine with EMBEDDINGS_MODE=openai")
        print("3. Vector search will gracefully fall back to text search due to dimension mismatch")
        print("4. Or re-index your data with OpenAI embeddings for full vector search")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

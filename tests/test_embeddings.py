#!/usr/bin/env python3
"""
Test script for embedding functionality.
Tests HF Inference API without requiring a full Elasticsearch setup.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_hf_embeddings():
    """Test HF Inference API embedding generation."""
    print("🧪 Testing HF Inference API embeddings...")
    
    # Set up environment for HF mode
    os.environ["EMBEDDINGS_MODE"] = "hf"
    
    # Check if HF token is set
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token or hf_token == "your_hf_token_here":
        print("❌ HF_API_TOKEN not set or using placeholder value")
        print("Please set a real Hugging Face token:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new token")
        print("3. Set: export HF_API_TOKEN=your_actual_token")
        return False
    
    try:
        from embeddings import get_query_embedding
        
        # Test with a sample query
        test_query = "What are the best cloud providers for AI workloads?"
        print(f"Query: {test_query}")
        
        embedding = get_query_embedding(test_query)
        
        print(f"✅ Success! Generated embedding:")
        print(f"   - Dimensions: {len(embedding)}")
        print(f"   - First 5 values: {embedding[:5]}")
        print(f"   - Expected dims: 384 (for sentence-transformers/all-MiniLM-L6-v2)")
        
        if len(embedding) == 384:
            print("✅ Dimensions match expected 384!")
        else:
            print(f"⚠️  Unexpected dimensions: {len(embedding)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're on the feature/openai-embeddings branch")
        return False
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        print(f"Error type: {type(e)}")
        return False

def test_openai_embeddings():
    """Test OpenAI embedding generation."""
    print("\n🧪 Testing OpenAI embeddings...")
    
    os.environ["EMBEDDINGS_MODE"] = "openai"
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY not set or using placeholder")
        print("Skipping OpenAI test...")
        return False
    
    try:
        from embeddings import get_query_embedding
        
        test_query = "Compare AWS and GCP for machine learning"
        print(f"Query: {test_query}")
        
        embedding = get_query_embedding(test_query)
        
        print(f"✅ Success! Generated OpenAI embedding:")
        print(f"   - Dimensions: {len(embedding)}")
        print(f"   - First 5 values: {embedding[:5]}")
        print(f"   - Expected dims: 1536 (for text-embedding-3-small)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating OpenAI embedding: {e}")
        return False

def test_dimension_validation():
    """Test dimension validation logic."""
    print("\n🧪 Testing dimension validation...")
    
    try:
        from embeddings import get_query_embedding
        
        # Test with expected dimensions
        os.environ["EMBEDDINGS_MODE"] = "hf"
        embedding = get_query_embedding("test query", expected_dims=384)
        print(f"✅ HF embedding validation passed: {len(embedding)} dims")
        
        # Test dimension mismatch (should raise ValueError)
        try:
            os.environ["EMBEDDINGS_MODE"] = "openai"
            embedding = get_query_embedding("test query", expected_dims=384)
            print("❌ Expected ValueError for dimension mismatch!")
            return False
        except ValueError as e:
            print(f"✅ Dimension validation works: {e}")
            return True
            
    except Exception as e:
        print(f"❌ Error in dimension validation test: {e}")
        return False

def main():
    """Run all embedding tests."""
    print("🔬 Embedding System Test Suite")
    print("=" * 50)
    
    # Show current environment
    print(f"Current branch: ", end="")
    os.system("git branch --show-current")
    print(f"EMBEDDINGS_MODE: {os.getenv('EMBEDDINGS_MODE', 'not set')}")
    print(f"MODEL_NAME: {os.getenv('MODEL_NAME', 'not set')}")
    print()
    
    results = []
    
    # Test HF embeddings
    results.append(("HF Embeddings", test_hf_embeddings()))
    
    # Test OpenAI embeddings (if configured)
    results.append(("OpenAI Embeddings", test_openai_embeddings()))
    
    # Test dimension validation
    results.append(("Dimension Validation", test_dimension_validation()))
    
    # Summary
    print("\n📊 Test Results:")
    print("=" * 30)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("🎉 All tests passed! Ready for deployment.")
    else:
        print("⚠️  Some tests failed. Check configuration.")
        
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

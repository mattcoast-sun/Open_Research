#!/usr/bin/env python3
"""
Demo script showing multi-model embedding search capabilities.

This script demonstrates:
1. Uploading data with multiple embedding models
2. Searching with different models
3. Comparing search results between models
4. Using ensemble methods
"""

import os
import sys
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elasticsearch import Elasticsearch
from embeddings_multi_model import (
    get_available_models, 
    search_with_model, 
    search_with_multiple_models,
    get_model_info
)

# Import Elasticsearch configuration from upload script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import upload_data_multi_model

# Use the same Elasticsearch client as the upload script
es = upload_data_multi_model.es

# Index names - use production indices
CLOUDS_INDEX = "cloud_providers"  
BEST_PRACTICES_INDEX = "cloud_best_practices"


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_results(results: List[Dict[str, Any]], metadata: Dict[str, Any], max_results: int = 3):
    """Print search results in a formatted way."""
    print(f"Found {len(results)} results (showing top {min(max_results, len(results))})")
    print(f"Metadata: {metadata}")
    
    for i, result in enumerate(results[:max_results]):
        print(f"\n--- Result {i+1} ---")
        print(f"ID: {result['document_id']}")
        print(f"Score: {result['similarity_score']:.4f}")
        
        source = result['source']
        title = source.get('provider_name') or source.get('title', 'Unknown')
        print(f"Title: {title}")
        
        # Show content snippet
        content = source.get('overview') or source.get('summary', '')
        if content:
            snippet = content[:200] + "..." if len(content) > 200 else content
            print(f"Content: {snippet}")
        
        # Show individual scores if available (ensemble results)
        if 'individual_scores' in result:
            print(f"Individual scores: {result['individual_scores']}")


def demo_model_info():
    """Demonstrate model information retrieval."""
    print_section("Model Information")
    
    # Show all available models
    all_models = get_model_info()
    print("Configured models:")
    for model_key, config in all_models["available_models"].items():
        print(f"  {model_key}: {config['model_name']} ({config['dims']} dims)")
    
    # Show specific model info
    print(f"\nDetails for model 'v1':")
    v1_info = get_model_info("v1")
    print(f"  Type: {v1_info['type']}")
    print(f"  Model: {v1_info['model_name']}")
    print(f"  Dimensions: {v1_info['dims']}")
    print(f"  Vector field: {v1_info['vector_field']}")


def demo_available_models():
    """Demonstrate checking available models in indices."""
    print_section("Available Models in Indices")
    
    try:
        clouds_models = get_available_models(es, CLOUDS_INDEX)
        print(f"Cloud providers index models: {clouds_models}")
    except Exception as e:
        print(f"Could not check cloud providers index: {e}")
    
    try:
        practices_models = get_available_models(es, BEST_PRACTICES_INDEX)
        print(f"Best practices index models: {practices_models}")
    except Exception as e:
        print(f"Could not check best practices index: {e}")


def demo_single_model_search():
    """Demonstrate searching with a single model."""
    print_section("Single Model Search")
    
    query = "cost optimization strategies for AWS"
    
    # Try different models
    models_to_try = ["v1", "v2", "v3"]
    
    for model_key in models_to_try:
        print(f"\n--- Searching with model {model_key} ---")
        try:
            results, metadata = search_with_model(
                es, BEST_PRACTICES_INDEX, query, model_key, size=3
            )
            print_results(results, metadata, max_results=2)
        except Exception as e:
            print(f"Error searching with {model_key}: {e}")


def demo_ensemble_search():
    """Demonstrate ensemble search with multiple models."""
    print_section("Ensemble Search")
    
    query = "serverless computing advantages"
    
    try:
        # Get available models first
        available_models = get_available_models(es, CLOUDS_INDEX)
        print(f"Using models: {available_models}")
        
        if len(available_models) < 2:
            print("Need at least 2 models for ensemble search. Upload data with multiple models first.")
            return
        
        # Try different ensemble methods
        ensemble_methods = ["average", "max", "weighted"]
        
        for method in ensemble_methods:
            print(f"\n--- Ensemble method: {method} ---")
            try:
                results, metadata = search_with_multiple_models(
                    es, CLOUDS_INDEX, query, 
                    model_keys=available_models[:2],  # Use first 2 available models
                    size=3, 
                    ensemble_method=method
                )
                print_results(results, metadata, max_results=2)
            except Exception as e:
                print(f"Error with ensemble method {method}: {e}")
                
    except Exception as e:
        print(f"Error in ensemble search: {e}")


def demo_comparison():
    """Demonstrate comparing results across different models."""
    print_section("Model Comparison")
    
    query = "security best practices for cloud infrastructure"
    
    try:
        available_models = get_available_models(es, BEST_PRACTICES_INDEX)
        
        if len(available_models) < 2:
            print("Need at least 2 models for comparison. Upload data with multiple models first.")
            return
        
        print(f"Comparing models: {available_models[:2]}")
        
        all_results = {}
        
        # Search with each model
        for model_key in available_models[:2]:
            try:
                results, metadata = search_with_model(
                    es, BEST_PRACTICES_INDEX, query, model_key, size=5
                )
                all_results[model_key] = {
                    "results": results,
                    "metadata": metadata
                }
                print(f"\nModel {model_key} found {len(results)} results")
            except Exception as e:
                print(f"Error with model {model_key}: {e}")
        
        # Compare top results
        if len(all_results) >= 2:
            print(f"\n--- Top result comparison ---")
            for model_key, data in all_results.items():
                if data["results"]:
                    top_result = data["results"][0]
                    print(f"\nModel {model_key} top result:")
                    print(f"  Score: {top_result['similarity_score']:.4f}")
                    print(f"  ID: {top_result['document_id']}")
                    
                    source = top_result['source']
                    title = source.get('title', source.get('provider_name', 'Unknown'))
                    print(f"  Title: {title}")
        
    except Exception as e:
        print(f"Error in comparison: {e}")


def main():
    """Run all demonstrations."""
    print("Multi-Model Embedding Search Demo")
    print("=" * 60)
    
    # Check Elasticsearch connection
    try:
        if not es.ping():
            print("ERROR: Cannot connect to Elasticsearch")
            return
        print("✓ Connected to Elasticsearch")
    except Exception as e:
        print(f"ERROR: Elasticsearch connection failed: {e}")
        return
    
    # Run demonstrations
    demo_model_info()
    demo_available_models()
    demo_single_model_search()
    demo_ensemble_search()
    demo_comparison()
    
    print_section("Demo Complete")
    print("To get started with multi-model embeddings:")
    print("1. Set OPENAI_API_KEY in your environment")
    print("2. Run: ACTIVE_MODELS='v1,v2' python upload_data_multi_model.py")
    print("3. Use the new search functions in your applications")


if __name__ == "__main__":
    main()

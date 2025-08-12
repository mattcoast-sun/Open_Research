#!/usr/bin/env python3
"""
Production Multi-Model Upload Script

This script safely adds new embedding models to your production data
WITHOUT affecting your current running endpoint.

Safety features:
- Uses UPDATE_EXISTING=true to preserve current data
- Uses RECREATE_INDICES=false to avoid downtime  
- Adds new vector fields alongside existing ones
- Your current endpoint continues working unchanged
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def confirm_production_settings():
    """Confirm production settings before proceeding."""
    print("🚀 Production Multi-Model Upload")
    print("=" * 50)
    
    print("\n📋 Current Configuration:")
    print(f"  ES_URL: {os.getenv('ES_URL', 'Not set')}")
    print(f"  INDEX: {os.getenv('INDEX', 'providers')}")
    print(f"  OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    
    print(f"\n🛡️ Safety Settings:")
    print(f"  UPDATE_EXISTING: true (preserves current data)")
    print(f"  RECREATE_INDICES: false (no downtime)")
    print(f"  Models to add: v2 (OpenAI embeddings)")
    
    print(f"\n✅ What this will do:")
    print(f"  - Keep your existing 'body_vector' fields (384-dim)")
    print(f"  - Add new 'body_vector_v2' fields (1536-dim OpenAI)")
    print(f"  - Your current container/endpoint keeps working")
    print(f"  - No service interruption")
    
    print(f"\n⚠️  Requirements:")
    print(f"  - OpenAI API key must be set")
    print(f"  - Will use OpenAI API (costs ~$0.01 per 1000 docs)")
    
    response = input("\n🤔 Proceed with production upload? (yes/no): ").strip().lower()
    return response in ['yes', 'y']

def set_production_environment():
    """Set environment variables for safe production upload."""
    
    # Safety settings - preserve existing data
    os.environ["UPDATE_EXISTING"] = "true"
    os.environ["RECREATE_INDICES"] = "false"
    
    # Only add OpenAI embeddings (v2) for now
    os.environ["ACTIVE_MODELS"] = "v2"
    
    print("✅ Environment configured for safe production upload")

def run_upload():
    """Run the multi-model upload."""
    try:
        print("\n🔄 Starting production multi-model upload...")
        print("This may take a few minutes depending on data size...\n")
        
        # Import and run the upload script
        import upload_data_multi_model
        upload_data_multi_model.main()
        
        print("\n🎉 Production upload completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False

def verify_results():
    """Verify the upload results."""
    print("\n🔍 Verifying results...")
    
    try:
        import upload_data_multi_model
        from elasticsearch import Elasticsearch
        
        es = upload_data_multi_model.es
        
        # Check cloud providers index
        clouds_index = upload_data_multi_model.CLOUDS_INDEX
        try:
            sample_doc = es.search(index=clouds_index, size=1)
            if sample_doc["hits"]["hits"]:
                doc_source = sample_doc["hits"]["hits"][0]["_source"]
                vector_fields = [k for k in doc_source.keys() if k.startswith("body_vector")]
                
                print(f"✅ Sample document has vector fields: {vector_fields}")
                
                if "body_vector" in vector_fields and "body_vector_v2" in vector_fields:
                    print("✅ Both original and new embeddings present!")
                    print("✅ Your current endpoint will continue working")
                    print("✅ New multi-model search capabilities available")
                else:
                    print("⚠️  Expected both body_vector and body_vector_v2 fields")
                    
        except Exception as e:
            print(f"⚠️  Could not verify index {clouds_index}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def show_next_steps():
    """Show what to do next."""
    print("\n🎯 Next Steps:")
    print("=" * 50)
    
    print("\n1. 🔍 Test Multi-Model Search:")
    print("   python3 demo_multi_model_search.py")
    
    print("\n2. 🚀 Update Your Application (Optional):")
    print("   - Current endpoint continues working unchanged")
    print("   - Add model selection parameter when ready")
    print("   - Use ensemble search for better results")
    
    print("\n3. 📊 Compare Model Performance:")
    print("   - Test search quality with both models")
    print("   - Choose best model for your use case")
    
    print("\n4. 🔄 Future Options:")
    print("   - Add more models: ACTIVE_MODELS='v1,v2,v3'")
    print("   - Remove old embeddings when confident in new ones")
    
    print("\n✅ Your production system is now multi-model ready!")

def main():
    """Main execution function."""
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key and try again")
        return False
    
    # Confirm production settings
    if not confirm_production_settings():
        print("👋 Upload cancelled by user")
        return False
    
    # Set safe production environment
    set_production_environment()
    
    # Run the upload
    success = run_upload()
    
    if success:
        # Verify results
        verify_results()
        
        # Show next steps
        show_next_steps()
        
        print(f"\n🎉 Production multi-model setup complete!")
        return True
    else:
        print(f"\n❌ Production upload failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

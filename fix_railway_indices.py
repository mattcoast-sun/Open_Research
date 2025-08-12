#!/usr/bin/env python3
"""
Fix Railway Elasticsearch Indices

This script creates the missing Elasticsearch indices on Railway deployment.
It uploads the necessary data files to create the required indices.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_data_files():
    """Check if required data files exist"""
    print("📁 Checking Data Files...")
    print("=" * 40)
    
    required_files = [
        "clouds.json",
        "cloud_best_practices.json", 
        "innovative_ideas.json",
        "tech_trends_innovation.json"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_name in required_files:
        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name)
            print(f"✅ {file_name} - {file_size:,} bytes")
            existing_files.append(file_name)
        else:
            print(f"❌ {file_name} - NOT FOUND")
            missing_files.append(file_name)
    
    return existing_files, missing_files

def set_upload_environment():
    """Set environment variables for uploading"""
    print("\n🔧 Setting Upload Environment...")
    print("=" * 40)
    
    # Set variables for fresh upload
    os.environ["UPDATE_EXISTING"] = "false"
    os.environ["RECREATE_INDICES"] = "true"
    os.environ["ACTIVE_MODELS"] = "v1,v2"  # Use both models if OpenAI key available
    
    print("✅ UPDATE_EXISTING = false")
    print("✅ RECREATE_INDICES = true") 
    print("✅ ACTIVE_MODELS = v1,v2")
    
    # Check OpenAI key
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY found - will use OpenAI embeddings")
    else:
        print("⚠️ OPENAI_API_KEY not found - will use sentence-transformers only")
        os.environ["ACTIVE_MODELS"] = "v1"

def run_upload():
    """Run the upload script to create indices"""
    print("\n🚀 Running Upload Script...")
    print("=" * 40)
    
    try:
        # Try multi-model upload first
        if os.path.exists("upload_data_multi_model.py"):
            print("Using multi-model upload script...")
            import upload_data_multi_model
            upload_data_multi_model.main()
        elif os.path.exists("upload_data_new.py"):
            print("Using standard upload script...")
            import upload_data_new
            # Assume it has a main function or run directly
            exec(open("upload_data_new.py").read())
        else:
            print("❌ No upload script found!")
            return False
        
        print("✅ Upload completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def verify_indices():
    """Verify that indices were created"""
    print("\n🔍 Verifying Indices...")
    print("=" * 40)
    
    try:
        # Import the test script
        import test_elasticsearch_indices
        
        # Test connection
        es = test_elasticsearch_indices.test_elasticsearch_connection()
        if not es:
            print("❌ Cannot connect to Elasticsearch")
            return False
        
        # Check indices
        existing_indices, missing_indices = test_elasticsearch_indices.check_indices(es)
        
        if not missing_indices:
            print("🎉 All indices created successfully!")
            return True
        else:
            print(f"⚠️ Still missing indices: {missing_indices}")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Railway Elasticsearch Indices Fix")
    print("=" * 50)
    
    # Check data files
    existing_files, missing_files = check_data_files()
    
    if missing_files:
        print(f"\n❌ Missing data files: {missing_files}")
        print("Please ensure all data files are present before running this script.")
        return False
    
    # Set environment
    set_upload_environment()
    
    # Run upload
    success = run_upload()
    if not success:
        print("\n❌ Upload failed - cannot continue")
        return False
    
    # Verify results
    verification_success = verify_indices()
    
    if verification_success:
        print("\n🎉 Railway indices fix completed successfully!")
        print("Your API should now work properly.")
    else:
        print("\n⚠️ Fix may not be complete - check logs for issues")
    
    return verification_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Fix Missing Cloud Provider Indices

This script specifically creates the missing cloud provider indices:
- cloud_providers_providers 
- cloud_providers_best_practices

These are the indices causing the 404 errors on Railway.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_required_files():
    """Check if the required cloud data files exist"""
    print("📁 Checking Required Cloud Data Files...")
    print("=" * 50)
    
    required_files = {
        "clouds.json": "cloud_providers_providers",
        "cloud_best_practices.json": "cloud_providers_best_practices"
    }
    
    existing_files = {}
    missing_files = []
    
    for file_name, index_name in required_files.items():
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    doc_count = len(data) if isinstance(data, list) else 1
                    print(f"✅ {file_name} → {index_name} ({doc_count} documents)")
                    existing_files[file_name] = index_name
            except Exception as e:
                print(f"⚠️ {file_name} exists but has error: {e}")
                missing_files.append(file_name)
        else:
            print(f"❌ {file_name} → {index_name} (NOT FOUND)")
            missing_files.append(file_name)
    
    return existing_files, missing_files

def run_cloud_upload():
    """Upload only the cloud provider data"""
    print("\n🚀 Uploading Cloud Provider Data...")
    print("=" * 50)
    
    # Set environment for targeted upload
    os.environ["UPDATE_EXISTING"] = "false"
    os.environ["RECREATE_INDICES"] = "false"  # Don't recreate existing indices
    
    try:
        # Check if we have the multi-model upload script
        if os.path.exists("upload_data_multi_model.py"):
            print("Using multi-model upload script...")
            
            # Import and configure for cloud data only
            import upload_data_multi_model
            
            # Override the main function to upload only cloud data
            print("Uploading cloud providers and best practices...")
            
            # This will create the missing indices
            upload_data_multi_model.main()
            
        elif os.path.exists("upload_data_new.py"):
            print("Using standard upload script...")
            import upload_data_new
            # Run the upload
            exec(open("upload_data_new.py").read())
            
        else:
            print("❌ No upload script found!")
            return False
        
        print("✅ Cloud data upload completed!")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"Error type: {type(e)}")
        return False

def verify_cloud_indices():
    """Verify the cloud indices were created"""
    print("\n🔍 Verifying Cloud Indices...")
    print("=" * 50)
    
    try:
        # Import elasticsearch setup from main
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the ES client configuration from main.py
        from main import es, CLOUDS_INDEX, BEST_PRACTICES_INDEX
        
        if not es:
            print("❌ Elasticsearch client not available")
            return False
        
        if not es.ping():
            print("❌ Cannot connect to Elasticsearch")
            return False
        
        print("✅ Connected to Elasticsearch")
        
        # Check the specific indices that were missing
        target_indices = {
            CLOUDS_INDEX: "Cloud Providers",
            BEST_PRACTICES_INDEX: "Best Practices"
        }
        
        all_good = True
        
        for index_name, description in target_indices.items():
            try:
                if es.indices.exists(index=index_name):
                    count_result = es.count(index=index_name)
                    doc_count = count_result.get('count', 0)
                    print(f"✅ {description} ({index_name}): {doc_count} documents")
                else:
                    print(f"❌ {description} ({index_name}): Still missing")
                    all_good = False
            except Exception as e:
                print(f"❌ {description} ({index_name}): Error - {e}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main function to fix missing cloud indices"""
    print("🔧 Fix Missing Cloud Provider Indices")
    print("=" * 60)
    print("Target indices:")
    print("  - cloud_providers_providers")
    print("  - cloud_providers_best_practices")
    print()
    
    # Check required files
    existing_files, missing_files = check_required_files()
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        print("Please ensure clouds.json and cloud_best_practices.json exist.")
        return False
    
    # Upload cloud data
    upload_success = run_cloud_upload()
    if not upload_success:
        print("\n❌ Upload failed - cannot continue")
        return False
    
    # Verify results
    verification_success = verify_cloud_indices()
    
    if verification_success:
        print("\n🎉 Cloud indices fix completed successfully!")
        print("Your Railway API should now work properly.")
        print("\nTest with:")
        print("  curl https://researcherlegend-production.up.railway.app/research-pipeline \\")
        print("    -X POST -H 'Content-Type: application/json' \\")
        print("    -d '{\"question\": \"What are the best cloud providers for AI?\", \"skip_clarification\": true}'")
    else:
        print("\n⚠️ Some indices may still be missing - check the output above")
    
    return verification_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

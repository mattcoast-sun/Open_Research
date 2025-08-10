#!/usr/bin/env python3
"""
Manual test of HF API with correct payload format.
You can paste your HF token here temporarily for testing.
"""

import requests
import os

def test_hf_api_manually():
    """Test HF API directly with the fixed payload."""
    
    # ⚠️ TEMPORARY: Put your HF token here for testing
    # Get one from: https://huggingface.co/settings/tokens
    HF_TOKEN = "hf_your_token_here"  # Replace with actual token
    
    if HF_TOKEN == "hf_your_token_here":
        print("❌ Please set a real HF token in the script for testing")
        return False
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    
    # Test with the new payload format
    payload = {
        "inputs": ["What are the best cloud providers for AI?"],
        "options": {"wait_for_model": True}
    }
    
    print(f"🧪 Testing HF API with URL: {api_url}")
    print(f"Payload: {payload}")
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.ok:
            result = response.json()
            print(f"✅ Success!")
            print(f"Response type: {type(result)}")
            if isinstance(result, list) and result:
                if isinstance(result[0], list):
                    dims = len(result[0])
                    print(f"Dimensions: {dims}")
                    print(f"First 5 values: {result[0][:5]}")
                    if dims == 384:
                        print("✅ Perfect! 384 dimensions match your index")
                    return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    success = test_hf_api_manually()
    if success:
        print("\n🎉 HF API works! Now set HF_API_TOKEN environment variable and run the full test.")
    else:
        print("\n⚠️  Fix the HF token and try again.")

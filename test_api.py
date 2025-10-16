#!/usr/bin/env python3
"""
Test script for the FastAPI server
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_models():
    """Test models endpoint"""
    print("\n" + "="*60)
    print("Testing Models Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_transcribe_file(audio_file_path):
    """Test transcription with file upload"""
    print("\n" + "="*60)
    print("Testing Transcription (File Upload)")
    print("="*60)
    
    with open(audio_file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'model': 'openai/whisper-large-v3',
            'batch_size': 24,
            'use_flash': True,
            'return_timestamps': True
        }
        
        response = requests.post(f"{BASE_URL}/transcribe", files=files, data=data)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))


def test_transcribe_url(audio_url):
    """Test transcription with URL"""
    print("\n" + "="*60)
    print("Testing Transcription (URL)")
    print("="*60)
    
    data = {
        'url': audio_url,
        'model': 'openai/whisper-large-v3',
        'batch_size': 24,
        'use_flash': True,
        'return_timestamps': True
    }
    
    response = requests.post(f"{BASE_URL}/transcribe/url", data=data)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("FastAPI Server Test Suite")
    print("Make sure the server is running on http://localhost:8000")
    
    # Test health
    try:
        test_health()
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test models
    try:
        test_models()
    except Exception as e:
        print(f"Models check failed: {e}")
    
    # Example: Test with file (uncomment and provide path)
    # test_transcribe_file("path/to/your/audio.mp3")
    
    # Example: Test with URL (uncomment and provide URL)
    # test_transcribe_url("https://example.com/audio.mp3")
    
    print("\n" + "="*60)
    print("Tests Complete!")
    print("="*60)

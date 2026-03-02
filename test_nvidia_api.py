#!/usr/bin/env python3
"""Test script for Nvidia API with DNS fix."""
import os
import sys

# Load DNS fix from model module
sys.path.insert(0, '.')
from assistant import model  # noqa: F401 - this patches socket

import requests

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

# Load API key from environment
api_key = os.getenv("NVIDIA_API_KEY", "").strip()
if not api_key:
    print("Error: NVIDIA_API_KEY not set in environment")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
    "model": "qwen/qwen3.5-397b-a17b",
    "messages": [{"role": "user", "content": "Hello, tell me a short joke."}],
    "max_tokens": 1024,
    "temperature": 0.60,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0,
    "repetition_penalty": 1,
    "stream": stream,
}

print(f"Testing Nvidia API...")
print(f"Model: {payload['model']}")
print(f"Stream: {stream}")
print("-" * 50)

try:
    response = requests.post(invoke_url, headers=headers, json=payload, stream=stream, timeout=30)
    response.raise_for_status()
    
    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        import json
        print(json.dumps(response.json(), indent=2))
    
    print("-" * 50)
    print("✓ Success!")
    
except requests.exceptions.ConnectionError as e:
    print(f"✗ Connection error: {e}")
    sys.exit(1)
except requests.exceptions.HTTPError as e:
    print(f"✗ HTTP error: {e}")
    print(f"Response: {e.response.text}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

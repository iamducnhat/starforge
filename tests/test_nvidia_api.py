#!/usr/bin/env python3
"""Optional integration test for Nvidia API with DNS patch enabled."""

from __future__ import annotations

import os
import sys

import pytest
import requests

# Load DNS fix from model module
sys.path.insert(0, ".")
from assistant import model  # noqa: F401 - importing applies socket patch


@pytest.mark.integration
def test_nvidia_api_stream_integration() -> None:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        pytest.skip("NVIDIA_API_KEY not set; skipping Nvidia integration test")

    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": "qwen/qwen3.5-397b-a17b",
        "messages": [{"role": "user", "content": "Hello, tell me a short joke."}],
        "max_tokens": 256,
        "temperature": 0.60,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0,
        "repetition_penalty": 1,
        "stream": True,
    }

    response = requests.post(
        invoke_url, headers=headers, json=payload, stream=True, timeout=30
    )
    response.raise_for_status()

    got_line = False
    for line in response.iter_lines():
        if line:
            got_line = True
            break
    assert got_line, "Nvidia stream returned no content lines"

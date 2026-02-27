import urllib.request
import json
import os
import sys

api_key = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key={api_key}"
headers = {"Content-Type": "application/json"}
data = {
    "contents": [{"role": "user", "parts": [{"text": "Write a 3 sentence poem"}]}],
    "generationConfig": {"temperature": 0.2}
}

req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode("utf-8"), method="POST")
try:
    with urllib.request.urlopen(req) as response:
        print("Success!")
        for i, line in enumerate(response):
            print(line.decode().strip())
            if i > 10: break
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}")
    print(e.read().decode())

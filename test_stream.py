import requests
import json

# Test streaming endpoint
doc_id = "74bf760c-9651-4ff9-89f5-bc545e0ca5d8"  # from previous test

print("=== Testing /ask/stream ===")
url = 'http://127.0.0.1:8000/ask/stream'

resp = requests.post(url, json={
    'doc_id': doc_id,
    'query': 'What is machine learning?',
    'model': 'groq',
    'top_k': 3
}, stream=True, timeout=60)

print(f"Status: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('content-type')}")

for line in resp.iter_lines(decode_unicode=True):
    if line:
        print(f"Received: {line}")
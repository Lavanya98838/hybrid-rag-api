import requests
import json

doc_id = '74bf760c-9651-4ff9-89f5-bc545e0ca5d8'

print('=== Testing /ask ===')
resp = requests.post('http://127.0.0.1:8000/ask', json={
    'doc_id': doc_id,
    'query': 'What is machine learning?',
    'model': 'groq',
    'top_k': 3
}, timeout=60)
print(f'Status: {resp.status_code}')
data = resp.json()
print(f'Answer: {data.get("answer", "")[:200]}...')
print(f'Chunks: {len(data.get("source_chunks", []))}')

print('\n=== Testing /ask/stream ===')
resp = requests.post('http://127.0.0.1:8000/ask/stream', json={
    'doc_id': doc_id,
    'query': 'What is deep learning?',
    'model': 'groq',
    'top_k': 3
}, stream=True, timeout=60)
print(f'Status: {resp.status_code}')
for line in resp.iter_lines(decode_unicode=True):
    if line and line.startswith('data: '):
        data = json.loads(line[6:])
        if data.get('type') == 'token':
            print(data.get('content', ''), end='', flush=True)
        elif data.get('type') == 'metadata':
            print(f'\n[METADATA] {len(data.get("source_chunks", []))} chunks')
        elif data.get('type') == 'done':
            print('\n[DONE]')
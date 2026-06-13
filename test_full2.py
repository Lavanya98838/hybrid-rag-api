import requests
import json
import fitz

# Create test PDF
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), 'Machine learning is a subset of AI that enables computers to learn from data without being explicitly programmed. Supervised learning uses labeled data to train models like linear regression and decision trees. Deep learning uses neural networks with multiple layers for complex pattern recognition. Natural language processing combines linguistics and machine learning for text understanding and generation.')
doc.save('test.pdf')
doc.close()

# Upload PDF
url = 'http://127.0.0.1:8000/upload'
with open('test.pdf', 'rb') as f:
    files = {'file': ('test.pdf', f, 'application/pdf')}
    resp = requests.post(url, files=files)
    print('Upload:', resp.status_code)
    data = resp.json()
    print(json.dumps(data, indent=2))
    doc_id = data['doc_id']

# Test /ask endpoint
print('\n=== Testing /ask ===')
resp = requests.post('http://127.0.0.1:8000/ask', json={
    'doc_id': doc_id,
    'query': 'What is machine learning?',
    'model': 'groq',
    'top_k': 3
}, timeout=60)
print(f'Status: {resp.status_code}')
data = resp.json()
print(f'Answer: {data.get("answer", "")[:300]}...')
print(f'Chunks: {len(data.get("source_chunks", []))}')

# Test /ask/stream endpoint
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

# Test DELETE endpoint
print('\n=== Testing DELETE ===')
resp = requests.delete(f'http://127.0.0.1:8000/documents/{doc_id}')
print(f'Status: {resp.status_code}')
print(resp.json())
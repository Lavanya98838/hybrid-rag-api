import requests
import json

doc_id = 'bfe289a4-ab1f-47d1-b973-59a225eda64e'
resp = requests.post('http://127.0.0.1:8000/ask', json={
    'doc_id': doc_id,
    'query': 'HRMS',
    'model': 'groq',
    'top_k': 10
}, timeout=60)
data = resp.json()
for chunk in data.get('source_chunks', []):
    print(f'Chunk {chunk["chunk_id"]}: {chunk["text"][:150]}...')
    print(f'Score: {chunk["score"]:.4f}')
    print('---')
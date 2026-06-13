import requests
import json

doc_id = 'bfe289a4-ab1f-47d1-b973-59a225eda64e'

queries = [
    'What is this document about?',
    'Human Resource Management System',
    'Dr. Jugendra Kumar Dongre',
    'Lavanya Hardas',
    'Master of Computer Application',
    'Devi Ahilya Vishwavidyalaya',
    'declaration certificate acknowledgement',
    'project guide supervision',
]

print('=' * 60)
print('TEST: 10th Sem Report PDF')
print('=' * 60)

for q in queries:
    resp = requests.post('http://127.0.0.1:8000/ask', json={
        'doc_id': doc_id,
        'query': q,
        'model': 'gemini',
        'top_k': 3
    }, timeout=60)
    data = resp.json()
    print(f'\nQuery: "{q}"')
    print(f'Status: {resp.status_code}')
    if 'answer' in data:
        ans = data["answer"][:200].encode('ascii', 'replace').decode('ascii')
        print(f'Answer: {ans}...')
    if 'source_chunks' in data:
        print(f'Chunks found: {len(data["source_chunks"])}')
        for i, chunk in enumerate(data['source_chunks']):
            print(f'  {i+1}. Score: {chunk["score"]:.4f} | Text: {chunk["text"][:100]}...')
    else:
        print(f'Response: {json.dumps(data, indent=2)}')
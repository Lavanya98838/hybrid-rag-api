import requests
import json

doc_id = '603da95a-27bd-4a16-a853-a79171843d17'

queries = [
    'What is machine learning?',
    'supervised learning algorithms',
    'deep learning neural networks',
    'NLP transformers BERT GPT',
    'model evaluation metrics accuracy precision recall',
    'MLOps deployment Docker CI/CD',
    'clustering k-means unsupervised',
]

print("=" * 60)
print("DAY 2 HYBRID RETRIEVAL TEST")
print("=" * 60)

for q in queries:
    resp = requests.post('http://127.0.0.1:8000/ask', json={
        'doc_id': doc_id,
        'query': q,
        'model': 'groq',
        'top_k': 3
    }, timeout=60)
    data = resp.json()
    print(f'\nQuery: "{q}"')
    print(f'Status: {resp.status_code}')
    if 'source_chunks' in data:
        print(f'Chunks found: {len(data["source_chunks"])}')
        for i, chunk in enumerate(data['source_chunks']):
            print(f'  {i+1}. Score: {chunk["score"]:.4f} | Text: {chunk["text"][:100]}...')
    else:
        print(f'Response: {json.dumps(data, indent=2)}')
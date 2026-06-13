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

# Test asking questions
print('\n--- Testing /ask ---')
queries = [
    'What is machine learning?',
    'supervised learning algorithms',
    'deep learning neural networks',
    'NLP transformers BERT GPT',
]
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
    if 'answer' in data:
        print(f'Answer: {data["answer"][:300]}...')
    if 'source_chunks' in data:
        print(f'Chunks: {len(data["source_chunks"])}')
        for i, chunk in enumerate(data['source_chunks']):
            print(f'  {i+1}. Score: {chunk["score"]:.4f} | {chunk["text"][:100]}...')
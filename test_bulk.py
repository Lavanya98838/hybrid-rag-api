import requests
import json
import fitz

# Create test PDFs
for i in range(3):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), f'Document {i+1}: Machine learning is a subset of AI. Document {i+1} discusses supervised learning, deep learning, and NLP.')
    doc.save(f'test{i+1}.pdf')
    doc.close()

# Bulk upload
url = 'http://127.0.0.1:8000/upload/bulk'
files = [
    ('files', (f'test{i+1}.pdf', open(f'test{i+1}.pdf', 'rb'), 'application/pdf'))
    for i in range(3)
]
resp = requests.post(url, files=files)
print('Bulk Upload:', resp.status_code)
print(json.dumps(resp.json(), indent=2))

# List documents
print('\n=== List Documents ===')
resp = requests.get('http://127.0.0.1:8000/documents')
print(f'Status: {resp.status_code}')
print(json.dumps(resp.json(), indent=2))
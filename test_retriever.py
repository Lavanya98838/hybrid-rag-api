from retriever import HybridRetriever

chunks = [
    {'chunk_id': 0, 'text': 'Machine learning is a subset of AI that enables computers to learn from data'},
    {'chunk_id': 1, 'text': 'Supervised learning uses labeled data to train models like linear regression'},
    {'chunk_id': 2, 'text': 'Deep learning uses neural networks with multiple layers for complex patterns'},
    {'chunk_id': 3, 'text': 'Natural language processing combines linguistics and ML for text understanding'},
    {'chunk_id': 4, 'text': 'Model evaluation uses metrics like accuracy precision recall F1 score'},
]

retriever = HybridRetriever()
retriever.build_indexes(chunks)

print('=== BM25 SEARCH ===')
bm25_results = retriever.bm25_search('neural networks deep learning', k=3)
for r in bm25_results:
    print(f'  Score: {r["score"]:.4f} | {r["text"]}')

print('\n=== VECTOR SEARCH ===')
vector_results = retriever.vector_search('neural networks deep learning', k=3)
for r in vector_results:
    print(f'  Score: {r["score"]:.4f} | {r["text"]}')

print('\n=== HYBRID (RRF) ===')
hybrid_results = retriever.search('neural networks deep learning', k=3)
for r in hybrid_results:
    print(f'  RRF Score: {r["rrf_score"]:.4f} | {r["text"]}')

print('\n=== Test different query ===')
hybrid_results2 = retriever.search('supervised learning labeled data', k=3)
for r in hybrid_results2:
    print(f'  RRF Score: {r["rrf_score"]:.4f} | {r["text"]}')
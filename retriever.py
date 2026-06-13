import pickle
import os
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class HybridRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.bm25 = None
        self.faiss_index = None
        self.chunk_texts = []
        self.chunk_metadata = []

    def build_bm25(self, chunks: List[Dict[str, Any]]) -> BM25Okapi:
        """Build BM25 index from chunks."""
        self.chunk_texts = [chunk["text"] for chunk in chunks]
        self.chunk_metadata = chunks
        tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self.bm25

    def build_faiss(self, chunks: List[Dict[str, Any]]) -> faiss.Index:
        """Build FAISS index from chunks using sentence-transformers embeddings."""
        self.chunk_texts = [chunk["text"] for chunk in chunks]
        self.chunk_metadata = chunks
        embeddings = self.embedding_model.encode(self.chunk_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        return self.faiss_index

    def build_indexes(self, chunks: List[Dict[str, Any]]) -> Tuple[BM25Okapi, faiss.Index]:
        """Build both BM25 and FAISS indexes."""
        self.build_bm25(chunks)
        self.build_faiss(chunks)
        return self.bm25, self.faiss_index

    def bm25_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25 keyword matching."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_bm25() first.")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "chunk_id": self.chunk_metadata[idx]["chunk_id"],
                    "text": self.chunk_texts[idx],
                    "score": float(scores[idx]),
                    "metadata": self.chunk_metadata[idx]
                })
        return results

    def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using FAISS vector similarity."""
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss() first.")
        
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "chunk_id": self.chunk_metadata[idx]["chunk_id"],
                    "text": self.chunk_texts[idx],
                    "score": float(score),
                    "metadata": self.chunk_metadata[idx]
                })
        return results

    def rrf_fusion(self, bm25_hits: List[Dict[str, Any]], vector_hits: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """Merge BM25 and vector search results using Reciprocal Rank Fusion."""
        rrf_scores = {}
        
        for rank, hit in enumerate(bm25_hits):
            chunk_id = hit["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"score": 0.0, "hit": hit}
            rrf_scores[chunk_id]["score"] += 1.0 / (k + rank + 1)
        
        for rank, hit in enumerate(vector_hits):
            chunk_id = hit["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"score": 0.0, "hit": hit}
            rrf_scores[chunk_id]["score"] += 1.0 / (k + rank + 1)
        
        fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        
        results = []
        for item in fused:
            hit = item["hit"].copy()
            hit["rrf_score"] = item["score"]
            results.append(hit)
        
        return results

    def search(self, query: str, k: int = 5, fusion_k: int = 60) -> List[Dict[str, Any]]:
        """Hybrid search: BM25 + Vector + RRF fusion."""
        bm25_hits = self.bm25_search(query, k=k)
        vector_hits = self.vector_search(query, k=k)
        return self.rrf_fusion(bm25_hits, vector_hits, k=fusion_k)

    def save(self, path: str):
        """Save indexes to disk."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
        faiss.write_index(self.faiss_index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({"texts": self.chunk_texts, "metadata": self.chunk_metadata}, f)

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2") -> "HybridRetriever":
        """Load indexes from disk."""
        retriever = cls(model_name)
        with open(os.path.join(path, "bm25.pkl"), "rb") as f:
            retriever.bm25 = pickle.load(f)
        retriever.faiss_index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
            retriever.chunk_texts = data["texts"]
            retriever.chunk_metadata = data["metadata"]
        return retriever


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    retriever = HybridRetriever()
    return retriever.build_bm25(chunks)


def bm25_search(query: str, k: int = 5, retriever: HybridRetriever = None) -> List[Dict[str, Any]]:
    if retriever is None:
        raise ValueError("Pass a HybridRetriever instance")
    return retriever.bm25_search(query, k)


def build_faiss(chunks: List[Dict[str, Any]]) -> faiss.Index:
    retriever = HybridRetriever()
    return retriever.build_faiss(chunks)


def vector_search(query: str, k: int = 5, retriever: HybridRetriever = None) -> List[Dict[str, Any]]:
    if retriever is None:
        raise ValueError("Pass a HybridRetriever instance")
    return retriever.vector_search(query, k)


def rrf_fusion(bm25_hits: List[Dict[str, Any]], vector_hits: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    retriever = HybridRetriever()
    return retriever.rrf_fusion(bm25_hits, vector_hits, k)
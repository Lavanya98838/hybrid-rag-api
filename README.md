# Hybrid Document Q&A API

Upload PDFs, ask questions in plain English, get AI-powered answers using **BM25 + FAISS hybrid search** with **Groq** or **Gemini**.

## Architecture

```
PDF Upload → S3 Storage → Text Extraction → Chunking (500 words, 50 overlap)
                                                    ↓
                                           ┌──────────────────┐
                                           │  HybridRetriever │
                                           │  ┌────────────┐  │
                                           │  │ BM25 (keyword)│  │
                                           │  ├────────────┤  │
                                           │  │ FAISS (vector) │  │
                                           │  └────────────┘  │
                                           │  + RRF Fusion    │
                                           └──────────────────┘
                                                    ↓
                                           ┌──────────────────┐
                                           │  LLM (Groq/Gemini)│
                                           │  llama-3.3-70b   │
                                           │  gemini-1.5-flash │
                                           └──────────────────┘
                                                    ↓
                                              Answer + Sources
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload PDF (max 10MB) |
| POST | `/upload/bulk` | Upload multiple PDFs |
| GET | `/documents` | List uploaded documents |
| DELETE | `/documents/{doc_id}` | Delete a document |
| POST | `/ask` | Ask a question (returns answer + source chunks) |
| POST | `/ask/stream` | Ask with SSE streaming |

## Quick Start

```bash
# Clone & install
git clone <repo-url> && cd hybrid_rag_api
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set up .env
cp .env.example .env   # Fill in your API keys

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t hybrid-rag-api .
docker run -p 8080:8080 hybrid-rag-api
```

## GCP Cloud Run Deploy

```bash
# One-time setup
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build & deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/hybrid-rag-api
gcloud run deploy hybrid-rag-api \
  --image gcr.io/YOUR_PROJECT_ID/hybrid-rag-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="AWS_ACCESS_KEY_ID=xxx,AWS_SECRET_ACCESS_KEY=xxx,AWS_REGION=us-east-1,S3_BUCKET_NAME=hybrid-rag-lavanya,GROQ_API_KEY=xxx,GEMINI_API_KEY=xxx"
```

## Tech Stack

- **Framework:** FastAPI (Python 3.11)
- **Search:** BM25 (rank-bm25) + FAISS (sentence-transformers all-MiniLM-L6-v2) + RRF fusion
- **LLMs:** Groq (llama-3.3-70b-versatile), Gemini (gemini-1.5-flash)
- **Storage:** AWS S3
- **Cache:** Redis (optional)
- **Auth:** API key authentication with rate limiting (slowapi)
- **Deployment:** Docker, GCP Cloud Run

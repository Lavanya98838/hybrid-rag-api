import os
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from storage import upload_to_s3, download_from_s3, delete_from_s3
from extractor import extract_and_chunk
from retriever import HybridRetriever
from llm import generate_answer, generate_answer_stream

load_dotenv()

app = FastAPI(
    title="Hybrid Document Q&A API",
    description="Upload any PDF, ask questions in plain English, get AI-powered answers using BM25 + FAISS hybrid search with Groq or Gemini.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_DIR = Path("./indexes")
INDEX_DIR.mkdir(exist_ok=True)

# In-memory store for doc metadata and retrievers
# Format: { doc_id: { "s3_key": ..., "filename": ..., "chunks": [...], "retriever": HybridRetriever, "index_path": ... } }
document_store: dict = {}

MAX_FILE_SIZE_MB = 10


def load_existing_indexes():
    """Load all persisted indexes on startup."""
    for index_path in INDEX_DIR.iterdir():
        if index_path.is_dir() and (index_path / "faiss.index").exists():
            doc_id = index_path.name
            try:
                retriever = HybridRetriever.load(str(index_path))

                # Try to restore filename from saved info, fall back to doc_id
                info_path = index_path / "info.json"
                filename = doc_id
                if info_path.exists():
                    import json
                    info = json.loads(info_path.read_text())
                    filename = info.get("filename", doc_id)

                document_store[doc_id] = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "retriever": retriever,
                    "chunks": retriever.chunk_metadata,
                    "total_chunks": len(retriever.chunk_metadata),
                    "index_path": str(index_path),
                }
                print(f"Loaded index for doc_id: {doc_id} ({filename})")
            except Exception as e:
                print(f"Failed to load index {doc_id}: {e}")


@app.on_event("startup")
def startup_event():
    load_existing_indexes()


# ─────────────────────────────────────────────
# Serve Frontend
# ─────────────────────────────────────────────

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("frontend/index.html")


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/health", tags=["Health"])
@app.head("/health", tags=["Health"], include_in_schema=False)
def root():
    return {"status": "ok", "message": "Hybrid RAG API is running 🚀"}


# ─────────────────────────────────────────────
# POST /upload — Upload a PDF to S3 + chunk it
# ─────────────────────────────────────────────

@app.post("/upload", tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file.
    - Validates file type and size
    - Saves to AWS S3
    - Extracts text and splits into 500-word chunks with 50-word overlap
    - Returns doc_id (use this for /ask queries)
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    # Validate file size (10MB limit)
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum allowed is {MAX_FILE_SIZE_MB}MB."
        )

    # Upload to S3
    try:
        s3_result = upload_to_s3(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

    # Save PDF temporarily to disk for pdfplumber
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Extract + chunk
    try:
        chunks = extract_and_chunk(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")
    finally:
        os.unlink(tmp_path)  # always clean up temp file

    # Build retriever indexes
    retriever = HybridRetriever()
    retriever.build_indexes(chunks)

    # Persist indexes to disk
    doc_id = s3_result["doc_id"]
    index_path = INDEX_DIR / doc_id
    retriever.save(str(index_path))
    # Save document info for reload on restart
    import json
    info_path = index_path / "info.json"
    info_path.write_text(json.dumps({"filename": file.filename}))

    # Store in memory
    document_store[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "s3_key": s3_result["s3_key"],
        "file_url": s3_result["file_url"],
        "chunks": chunks,
        "total_chunks": len(chunks),
        "retriever": retriever,
        "index_path": str(index_path),
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "file_url": s3_result["file_url"],
        "total_chunks": len(chunks),
        "sample_chunk": chunks[0] if chunks else None,
        "message": "PDF uploaded, chunked, indexed, and persisted successfully. Use doc_id to ask questions.",
    }


# ─────────────────────────────────────────────
# GET /documents — List all uploaded docs
# ─────────────────────────────────────────────

@app.get("/documents", tags=["Documents"])
def list_documents():
    """
    List all uploaded documents in the current session.
    """
    docs = [
        {
            "doc_id": v["doc_id"],
            "filename": v["filename"],
            "total_chunks": v["total_chunks"],
        }
        for v in document_store.values()
    ]
    return {"total": len(docs), "documents": docs}


# ─────────────────────────────────────────────
# DELETE /documents/{doc_id} — Delete a document
# ─────────────────────────────────────────────

@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    """
    Delete a document from memory, disk, and S3.
    """
    if doc_id not in document_store:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    doc = document_store[doc_id]

    # Delete from S3
    try:
        delete_from_s3(doc["s3_key"])
    except Exception as e:
        print(f"S3 delete warning: {e}")

    # Delete local index
    index_path = doc.get("index_path")
    if index_path and os.path.exists(index_path):
        shutil.rmtree(index_path, ignore_errors=True)

    # Remove from memory
    del document_store[doc_id]

    return {"message": f"Document '{doc_id}' deleted successfully."}


# ─────────────────────────────────────────────
# POST /upload/bulk — Bulk upload multiple PDFs
# ─────────────────────────────────────────────

@app.post("/upload/bulk", tags=["Documents"])
async def upload_bulk(files: list[UploadFile] = File(...)):
    """
    Upload multiple PDF files at once.
    """
    results = []
    errors = []

    for file in files:
        try:
            if not file.filename.endswith(".pdf"):
                errors.append({"filename": file.filename, "error": "Only PDF files supported"})
                continue

            file_bytes = await file.read()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                errors.append({"filename": file.filename, "error": f"File too large: {file_size_mb:.1f}MB"})
                continue

            s3_result = upload_to_s3(file_bytes, file.filename)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                chunks = extract_and_chunk(tmp_path)
            except ValueError as e:
                errors.append({"filename": file.filename, "error": str(e)})
                os.unlink(tmp_path)
                continue
            except Exception as e:
                errors.append({"filename": file.filename, "error": f"PDF extraction failed: {str(e)}"})
                os.unlink(tmp_path)
                continue
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            retriever = HybridRetriever()
            retriever.build_indexes(chunks)

            doc_id = s3_result["doc_id"]
            index_path = INDEX_DIR / doc_id
            retriever.save(str(index_path))
            import json
            info_path = index_path / "info.json"
            info_path.write_text(json.dumps({"filename": file.filename}))

            document_store[doc_id] = {
                "doc_id": doc_id,
                "filename": file.filename,
                "s3_key": s3_result["s3_key"],
                "file_url": s3_result["file_url"],
                "chunks": chunks,
                "total_chunks": len(chunks),
                "retriever": retriever,
                "index_path": str(index_path),
            }

            results.append({
                "doc_id": doc_id,
                "filename": file.filename,
                "file_url": s3_result["file_url"],
                "total_chunks": len(chunks),
            })

        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    return {
        "successful": len(results),
        "failed": len(errors),
        "documents": results,
        "errors": errors,
    }


# ─────────────────────────────────────────────
# POST /ask — Ask a question (Day 3: retrieval + LLM)
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    doc_id: str
    query: str
    model: str = "cascade"  # "groq", "gemini", or "cascade"
    top_k: int = 5

@app.post("/ask", tags=["Q&A"])
def ask_question(request: AskRequest):
    """
    Ask a question about an uploaded document.
    Requires doc_id from /upload response.
    Returns LLM-generated answer + source chunks.
    """
    if request.doc_id not in document_store:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Please upload it first via POST /upload."
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.model not in ("groq", "gemini", "cascade"):
        raise HTTPException(status_code=400, detail="Model must be 'groq', 'gemini', or 'cascade'.")

    doc = document_store[request.doc_id]
    retriever = doc["retriever"]
    
    # Hybrid search: BM25 + FAISS + RRF
    results = retriever.search(request.query, k=request.top_k)
    
    source_chunks = [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"][:500] + ("..." if len(r["text"]) > 500 else ""),
            "score": r.get("rrf_score", r.get("score", 0)),
            "search_type": "hybrid"
        }
        for r in results
    ]
    
    # Generate LLM answer
    answer = generate_answer(request.query, results, request.model)

    return {
        "doc_id": request.doc_id,
        "query": request.query,
        "model": request.model,
        "answer": answer,
        "source_chunks": source_chunks,
    }


# ─────────────────────────────────────────────
# POST /ask/stream — Ask a question with streaming response
# ─────────────────────────────────────────────

class AskStreamRequest(BaseModel):
    doc_id: str
    query: str
    model: str = "cascade"  # "groq", "gemini", or "cascade"
    top_k: int = 5

@app.post("/ask/stream", tags=["Q&A"])
async def ask_question_stream(request: AskStreamRequest):
    """
    Ask a question about an uploaded document with streaming (SSE) response.
    Returns tokens as they are generated.
    """
    if request.doc_id not in document_store:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Please upload it first via POST /upload."
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.model not in ("groq", "gemini", "cascade"):
        raise HTTPException(status_code=400, detail="Model must be 'groq' or 'gemini'.")

    doc = document_store[request.doc_id]
    retriever = doc["retriever"]
    
    # Hybrid search: BM25 + FAISS + RRF
    results = retriever.search(request.query, k=request.top_k)
    
    source_chunks = [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"][:500] + ("..." if len(r["text"]) > 500 else ""),
            "score": r.get("rrf_score", r.get("score", 0)),
            "search_type": "hybrid"
        }
        for r in results
    ]
    
    async def generate():
        # Send source chunks first as metadata
        import json
        yield f"data: {json.dumps({'type': 'metadata', 'source_chunks': source_chunks})}\n\n"
        
        # Stream LLM tokens
        for token in generate_answer_stream(request.query, results, request.model):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
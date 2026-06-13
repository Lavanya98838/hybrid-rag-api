FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first to avoid pulling GPU/CUDA deps (saves ~1GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --retries 5 -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
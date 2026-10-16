# --- EVALS MICROSERVICE (RAGAS eval suite + Guardrails testing) ---
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Layer-cache: install deps before copying app code
COPY requirements-evals.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-evals.txt

# Copy eval code + the ingestion loaders that evals/app.py imports
COPY evals/ ./evals/
COPY app/ingestion/ ./app/ingestion/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["streamlit", "run", "evals/app.py", "--server.port=8080", "--server.address=0.0.0.0"]

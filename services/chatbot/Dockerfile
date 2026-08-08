FROM docker.io/python:3.11-slim

WORKDIR /app

# Imposta una cartella fissa per la cache di Chroma/HuggingFace
ENV XDG_CACHE_HOME=/app/.cache

COPY requirements.txt .
RUN pip install --trusted-host pypi.org --no-cache-dir -r requirements.txt

COPY . .

# Scarica il modello nella cartella fissa definita sopra
RUN python3 -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
"""
FantAssistant - Chatbot (Azure OpenAI GPT-4.1 + Chroma come servizio dedicato)
Retrieval semantico via un container Chroma separato (servizio "chroma" nel
compose), raggiunto via HTTP client. Chroma usa di default un modello di
embedding locale (scaricato automaticamente al primo utilizzo, gira via
ONNX su CPU) - quindi niente dipendenza da un deployment di embedding su
Azure.

Postgres resta la fonte di verita' per i dati strutturati (giocatori,
statistiche, rosa, asta, budget); Chroma e' solo l'indice per il retrieval
testuale usato dal chatbot.

Espone:
  POST /ingest  -> ricostruisce la collection Chroma a partire da Postgres
  POST /chat    -> fa una domanda, recupera i documenti piu' rilevanti da
                   Chroma, risponde via Azure OpenAI
  GET  /health  -> healthcheck
"""
import os

import httpx
import chromadb
import psycopg
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from psycopg.rows import dict_row
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI

DATABASE_URL = os.environ["DATABASE_URL"]
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

client_ai = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)
CHAT_DEPLOYMENT = os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4.1")


# Inizializziamo l'embedding function una sola volta fuori o dentro la funzione
_embedding_fn = None
_collection = None


def get_chroma_client():
    # Rimuove temporaneamente le variabili proxy per evitare che httpx le intercetti
    proxy_vars = {}
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        if key in os.environ:
            proxy_vars[key] = os.environ.pop(key)
            
    try:
        host = os.getenv("CHROMA_HOST", "chroma")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        
        return chromadb.HttpClient(host=host, port=port)
    finally:
        # Ripristina le variabili per le chiamate esterne (es. Azure OpenAI)
        os.environ.update(proxy_vars)


def get_collection():
    global _collection, _embedding_fn
    if _collection is None:
        client = get_chroma_client()
        
        # Istanziamo esplicitamente l'embedding function locale
        if _embedding_fn is None:
            _embedding_fn = ONNXMiniLM_L6_V2()
            
        _collection = client.get_or_create_collection(
            name="giocatori",
            embedding_function=_embedding_fn
        )
    return _collection


def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def chat_completion(prompt: str) -> str:
    resp = client_ai.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


app = FastAPI(title="FantAssistant Chatbot")


class ChatRequest(BaseModel):
    domanda: str
    top_k: int = 8


class ChatResponse(BaseModel):
    risposta: str
    contesto_usato: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    """
    Ricostruisce la collection Chroma a partire dai dati strutturati in
    Postgres. Un documento testuale per giocatore che riassume le sue info.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale,
                       s.stagione, s.presenze, s.gol, s.assist, s.media_voto,
                       mr.prezzo_pagato
                FROM giocatori g
                LEFT JOIN statistiche_storiche s ON s.giocatore_id = g.id
                LEFT JOIN mia_rosa mr ON mr.giocatore_id = g.id
            """)
            rows = cur.fetchall()

    if not rows:
        return {"documenti_indicizzati": 0}

    ids, documenti, metadati = [], [], []
    for r in rows:
        testo = (
            f"{r['nome']} ({r['ruolo']}, {r['squadra']}). "
            f"Quotazione attuale: {r['quotazione_attuale']}. "
        )
        if r["stagione"]:
            testo += (
                f"Stagione {r['stagione']}: {r['presenze']} presenze, "
                f"{r['gol']} gol, {r['assist']} assist, "
                f"media voto {r['media_voto']}. "
            )
        if r["prezzo_pagato"] is not None:
            testo += f"E' nella MIA ROSA, pagato {r['prezzo_pagato']} crediti."
        ids.append(str(r["id"]))
        documenti.append(testo)
        metadati.append({"giocatore_id": r["id"]})

    # upsert: aggiorna se l'id esiste gia', altrimenti crea. Cosi' un
    # /ingest ripetuto non duplica i documenti.
    get_collection().upsert(ids=ids, documents=documenti, metadatas=metadati)

    return {"documenti_indicizzati": len(ids)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    risultati = get_collection().query(query_texts=[req.domanda], n_results=req.top_k)
    contesto = risultati["documents"][0] if risultati["documents"] else []

    contesto_str = "\n".join(f"- {c}" for c in contesto)
    prompt = f"""Sei un assistente esperto di Fantacalcio. Rispondi alla domanda
usando SOLO le informazioni nel contesto sottostante. Se non trovi
informazioni sufficienti, dillo chiaramente invece di inventare.

Contesto:
{contesto_str}

Domanda: {req.domanda}

Risposta:"""

    risposta = chat_completion(prompt)

    return ChatResponse(risposta=risposta, contesto_usato=contesto)

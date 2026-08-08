"""
FantAssistant - Chatbot (Azure OpenAI GPT-4.1 + Chroma + SQL strutturato)
Due canali di retrieval, e il modello sceglie quale usare (function calling):

  1. Ricerca semantica su Chroma -> per domande aperte/fuzzy
     ("parlami di Lautaro", "chi assomiglia a...")
  2. Query SQL strutturata su Postgres -> per ordinamenti/filtri/top-N
     ("chi ha il FVM piu alto", "difensori sotto i 10 crediti", "quanti...")

Il secondo canale e' necessario perche' la ricerca vettoriale trova
documenti *semanticamente simili* alla domanda, non fa un vero ORDER BY:
per "top N per una colonna" la risposta corretta serve da SQL, non da RAG.

Espone:
  POST /ingest  -> ricostruisce la collection Chroma a partire da Postgres
  POST /chat    -> risponde usando lo strumento piu' adatto (o entrambi)
  GET  /health  -> healthcheck
"""
import os
import json

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

# Whitelist delle colonne ordinabili: mai interpolare l'input del modello
# direttamente in una query SQL, anche se "controllato" da noi via function
# calling - un mapping esplicito evita ogni rischio di injection.
COLONNE_ORDINABILI = {
    "fvm": "fvm",
    "quotazione_attuale": "quotazione_attuale",
    "quotazione_iniziale": "quotazione_iniziale",
    "nome": "nome",
}

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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cerca_giocatori_sql",
            "description": (
                "Cerca giocatori con filtri esatti e/o ordinamento su un "
                "campo numerico (FVM, quotazione). Usa questo strumento per "
                "domande tipo 'top N per X', 'chi ha il FVM piu alto', "
                "'difensori sotto i 10 crediti', 'quanti attaccanti ci sono "
                "sopra quota Y'. Non usarlo per domande generiche o su un "
                "singolo giocatore specifico gia' nominato."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ruolo": {
                        "type": "string",
                        "enum": ["P", "D", "C", "A"],
                        "description": "Filtra per ruolo. Omettere per tutti i ruoli.",
                    },
                    "squadra": {
                        "type": "string",
                        "description": "Filtra per squadra (match parziale). Omettere per tutte.",
                    },
                    "order_by": {
                        "type": "string",
                        "enum": list(COLONNE_ORDINABILI.keys()),
                        "description": "Campo su cui ordinare.",
                    },
                    "ordine": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "asc = dal piu basso, desc = dal piu alto.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Numero massimo di risultati (default 10, max 50).",
                    },
                },
                "required": ["order_by", "ordine"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ricerca_semantica",
            "description": (
                "Ricerca semantica su un riassunto testuale dei giocatori. "
                "Usa questo strumento per domande aperte o su un giocatore "
                "specifico nominato dall'utente (es. 'parlami di Lautaro', "
                "'chi ho in rosa', informazioni generali)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Testo da cercare."},
                    "top_k": {"type": "integer", "description": "Numero di risultati (default 8)."},
                },
                "required": ["query"],
            },
        },
    },
]


def esegui_cerca_giocatori_sql(ruolo=None, squadra=None, order_by="fvm", ordine="desc", limit=10):
    colonna = COLONNE_ORDINABILI.get(order_by, "fvm")
    direzione = "ASC" if ordine == "asc" else "DESC"
    limit = min(int(limit or 10), 50)

    query = f"""
        SELECT nome, ruolo, squadra, quotazione_attuale, fvm
        FROM giocatori
        WHERE 1=1
    """
    params = []
    if ruolo:
        query += " AND ruolo = %s"
        params.append(ruolo)
    if squadra:
        query += " AND squadra ILIKE %s"
        params.append(f"%{squadra}%")
    query += f" ORDER BY {colonna} {direzione} NULLS LAST LIMIT %s"
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def esegui_ricerca_semantica(query: str, top_k: int = 8):
    risultati = get_collection().query(query_texts=[query], n_results=top_k)
    return risultati["documents"][0] if risultati["documents"] else []


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
                SELECT g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale, g.fvm,
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
        if r["fvm"] is not None:
            testo += f"Fantavalore di mercato (FVM): {r['fvm']}. "
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
    messages = [
        {
            "role": "system",
            "content": (
                "Sei un assistente esperto di Fantacalcio. Hai due strumenti: "
                "uno per query strutturate (ordinamenti/filtri/top-N su dati "
                "esatti) e uno per ricerca semantica (domande aperte o su un "
                "giocatore specifico). Scegli quello giusto in base alla "
                "domanda - per 'chi ha il valore piu alto/basso' o 'top N' "
                "usa SEMPRE lo strumento SQL, mai la ricerca semantica. "
                "Rispondi SOLO in base ai risultati degli strumenti, mai "
                "inventando dati. Se i risultati non bastano, dillo."
            ),
        },
        {"role": "user", "content": req.domanda},
    ]

    contesto_usato: list[str] = []

    # Primo giro: il modello sceglie lo strumento (puo' anche chiamarne piu' di uno)
    resp = client_ai.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")

            if tool_call.function.name == "cerca_giocatori_sql":
                righe = esegui_cerca_giocatori_sql(**args)
                risultato = [dict(r) for r in righe]
                contesto_usato.extend(
                    f"{r['nome']} ({r['ruolo']}, {r['squadra']}): "
                    f"quot. {r['quotazione_attuale']}, FVM {r['fvm']}"
                    for r in risultato
                )
            elif tool_call.function.name == "ricerca_semantica":
                risultato = esegui_ricerca_semantica(
                    args.get("query", req.domanda), args.get("top_k", req.top_k)
                )
                contesto_usato.extend(risultato)
            else:
                risultato = []

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(risultato, default=str, ensure_ascii=False),
                }
            )

        # Secondo giro: il modello formula la risposta finale sui risultati ottenuti
        resp_finale = client_ai.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0.3,
        )
        risposta = resp_finale.choices[0].message.content
    else:
        # Il modello ha risposto direttamente senza usare strumenti
        # (capita per domande generiche non specifiche sui dati)
        risposta = msg.content

    return ChatResponse(risposta=risposta, contesto_usato=contesto_usato)
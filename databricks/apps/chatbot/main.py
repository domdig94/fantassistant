"""
FantAssistant Databricks — Chatbot App (FastAPI)

Sostituisce interamente services/chatbot/main.py.
Differenze rispetto alla versione Podman:
  - psycopg (PostgreSQL)  -> databricks-sql-connector
  - Chroma + ONNXMiniLM   -> Databricks Vector Search
  - Secrets da .env        -> env vars iniettate dal runtime Databricks App
  - Placeholder %s         -> ? (sintassi SQL Connector)
  - Nomi tabella           -> `catalog`.`schema`.tabella
"""
import os
import json
import logging
import time
import threading
from collections import deque
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.vector_search.client import VectorSearchClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Il WorkspaceClient si autentica automaticamente nel runtime Databricks App
# usando il service principal OAuth (DATABRICKS_CLIENT_ID + CLIENT_SECRET).
w = WorkspaceClient()
cfg = w.config
DATABRICKS_HOST = cfg.host
DATABRICKS_HTTP_PATH = os.environ["DATABRICKS_SQL_HTTP_PATH"]
CATALOG  = os.environ.get("UNITY_CATALOG", "fantassistant")
SCHEMA   = os.environ.get("UNITY_SCHEMA",  "main")
NS       = f"`{CATALOG}`.`{SCHEMA}`"

VS_ENDPOINT = os.environ.get("VECTOR_SEARCH_ENDPOINT", "fantassistant_vs_endpoint")
VS_INDEX    = f"{CATALOG}.{SCHEMA}.giocatori_vs_index"

client_ai = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)
CHAT_DEPLOYMENT = os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4.1")
MY_TEAM = os.environ.get("MY_TEAM", "Io")
ROSA_TARGET = {
    "P": int(os.environ.get("ROSA_TARGET_P", 3)),
    "D": int(os.environ.get("ROSA_TARGET_D", 8)),
    "C": int(os.environ.get("ROSA_TARGET_C", 8)),
    "A": int(os.environ.get("ROSA_TARGET_A", 6)),
}

# ---------------------------------------------------------------------------
# Persistent Logging (ring buffer + Delta table flush)
# ---------------------------------------------------------------------------
# NOTE: NS uses env vars that may not match the actual catalog/schema.
# Hardcode the log table to the known correct location.
LOG_TABLE = "`platform`.`fantassistant`.app_logs"
APP_NAME = os.environ.get("DATABRICKS_APP_NAME", "fantassistant-chatbot")
LOG_BUFFER: deque = deque(maxlen=500)
LOG_FLUSH_INTERVAL = 30  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fantassistant")


def _log_entry(level: str, message: str, extra: dict | None = None):
    """Append a structured log entry to the in-memory buffer."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        "extra": json.dumps(extra or {}, default=str, ensure_ascii=False),
    }
    LOG_BUFFER.append(entry)
    logger.log(getattr(logging, level, logging.INFO), message)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return dbsql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", ""),
        http_path=DATABRICKS_HTTP_PATH,
        credentials_provider=lambda: cfg.authenticate,
    )


def query_rows(sql: str, params: list | None = None) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def _flush_logs_to_table():
    """Flush buffered logs to the Delta table. Runs in background thread."""
    while True:
        time.sleep(LOG_FLUSH_INTERVAL)
        if not LOG_BUFFER:
            continue
        batch = list(LOG_BUFFER)
        LOG_BUFFER.clear()
        try:
            values = ",\n".join(
                "('{}', '{}', '{}', '{}', '{}')".format(
                    e["ts"],
                    e["level"],
                    e["message"].replace("'", "''"),
                    e["extra"].replace("'", "''"),
                    APP_NAME,
                )
                for e in batch
            )
            sql = f"INSERT INTO {LOG_TABLE} (ts, level, message, extra, app_name) VALUES {values}"
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
        except Exception as exc:
            logger.warning(f"Log flush failed: {exc}")
            for e in batch:
                LOG_BUFFER.append(e)


# Start background flush thread
_flush_thread = threading.Thread(target=_flush_logs_to_table, daemon=True)
_flush_thread.start()

# ---------------------------------------------------------------------------
# Vector Search (sostituisce Chroma)
# ---------------------------------------------------------------------------
_vs_client = None

def get_vs_client() -> VectorSearchClient:
    global _vs_client
    if _vs_client is None:
        _vs_client = VectorSearchClient(
            workspace_url=DATABRICKS_HOST,
            service_principal_client_id=os.environ.get("DATABRICKS_CLIENT_ID"),
            service_principal_client_secret=os.environ.get("DATABRICKS_CLIENT_SECRET"),
        )
    return _vs_client


def esegui_ricerca_semantica(query: str, top_k: int = 8) -> list[str]:
    vsc   = get_vs_client()
    index = vsc.get_index(VS_ENDPOINT, VS_INDEX)
    results = index.similarity_search(
        query_text=query,
        columns=["testo_embedding"],
        num_results=top_k,
    )
    cols = [c["name"] for c in results.get("result", {}).get("columns", [])]
    rows = results.get("result", {}).get("data_array", [])
    testo_idx = cols.index("testo_embedding") if "testo_embedding" in cols else 0
    return [row[testo_idx] for row in rows]

# ---------------------------------------------------------------------------
# Whitelist colonne (anti-injection)
# ---------------------------------------------------------------------------
COLONNE_ORDINABILI = {
    "fvm": "fvm",
    "quotazione_attuale": "quotazione_attuale",
    "quotazione_iniziale": "quotazione_iniziale",
    "nome": "nome",
}
COLONNE_STAT_ORDINABILI = {
    "fantamedia": "fantamedia",
    "media_voto": "media_voto",
    "gol": "gol",
    "assist": "assist",
    "presenze": "presenze",
    "ammonizioni": "ammonizioni",
    "espulsioni": "espulsioni",
    "rigori_parati": "rigori_parati",
    "autogol": "autogol",
}

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def esegui_cerca_giocatori_sql(
    ruolo=None, squadra=None, escludi_squadra=None,
    order_by="fvm", ordine="desc", limit=10, solo_liberi=True,
):
    colonna   = COLONNE_ORDINABILI.get(order_by, "fvm")
    direzione = "ASC" if ordine == "asc" else "DESC"
    limit     = min(int(limit or 10), 50)
    sql = f"""
        SELECT nome, ruolo, squadra, quotazione_attuale, fvm
        FROM {NS}.giocatori
        WHERE 1=1
    """
    params = []
    if ruolo:
        sql += " AND ruolo = ?"
        params.append(ruolo)
    if squadra:
        sql += " AND lower(squadra) LIKE lower(?)"
        params.append(f"%{squadra}%")
    if escludi_squadra:
        sql += " AND lower(squadra) NOT LIKE lower(?)"
        params.append(f"%{escludi_squadra}%")
    if solo_liberi:
        sql += f" AND id NOT IN (SELECT giocatore_id FROM {NS}.asta_log WHERE giocatore_id IS NOT NULL)"
    sql += f" ORDER BY {colonna} {direzione} NULLS LAST LIMIT {limit}"
    return query_rows(sql, params)


def esegui_statistiche_storiche(
    stagione=None, ruolo=None, order_by="fantamedia",
    ordine="desc", limit=10, nome=None,
):
    colonna   = COLONNE_STAT_ORDINABILI.get(order_by, "fantamedia")
    direzione = "ASC" if ordine == "asc" else "DESC"
    limit     = min(int(limit or 10), 50)
    sql = f"""
        SELECT g.nome, g.ruolo, g.squadra,
               s.stagione, s.presenze, s.media_voto, s.fantamedia,
               s.gol, s.assist, s.ammonizioni, s.espulsioni,
               s.gol_subiti, s.rigori_parati, s.autogol
        FROM {NS}.statistiche_storiche s
        JOIN {NS}.giocatori g ON g.id = s.giocatore_id
        WHERE 1=1
    """
    params = []
    if stagione:
        sql += " AND s.stagione = ?"
        params.append(stagione)
    if ruolo:
        sql += " AND g.ruolo = ?"
        params.append(ruolo)
    if nome:
        sql += " AND lower(g.nome) LIKE lower(?)"
        params.append(f"%{nome}%")
    sql += f" ORDER BY {colonna} {direzione} NULLS LAST LIMIT {limit}"
    return query_rows(sql, params)


def esegui_mio_budget():
    rows = query_rows(f"""
        SELECT s.nome, s.budget_totale,
               COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
        FROM {NS}.squadre s
        LEFT JOIN {NS}.asta_log a ON a.squadra_acquirente = s.nome
        WHERE lower(s.nome) = lower('{MY_TEAM}')
        GROUP BY s.nome, s.budget_totale
    """)
    if not rows:
        return {
            "errore": (
                f"Squadra '{MY_TEAM}' (da MY_TEAM) non trovata in squadre. "
                "Verifica il censimento con POST /squadre/init su auction-tracker."
            )
        }
    r = rows[0]
    return {
        "budget_totale":  float(r["budget_totale"]),
        "budget_speso":   float(r["budget_speso"]),
        "budget_residuo": float(r["budget_totale"]) - float(r["budget_speso"]),
    }


def esegui_mia_rosa():
    return query_rows(f"""
        SELECT g.nome, g.ruolo, g.squadra, mr.prezzo_pagato
        FROM {NS}.mia_rosa mr
        JOIN {NS}.giocatori g ON g.id = mr.giocatore_id
        ORDER BY g.ruolo, g.nome
    """)


def esegui_rosa_di_squadra(nome_squadra: str):
    parole = [p for p in nome_squadra.strip().split() if p]
    if not parole:
        return []
    condizioni, params = [], []
    for p in parole:
        condizioni.append("(lower(s.nome) LIKE lower(?) OR lower(s.allenatore) LIKE lower(?))")
        params.extend([f"%{p}%", f"%{p}%"])
    where_clause = " OR ".join(condizioni)
    return query_rows(f"""
        SELECT g.nome, g.ruolo, g.squadra, a.prezzo_finale,
               s.nome AS nome_squadra, s.allenatore
        FROM {NS}.asta_log a
        JOIN {NS}.giocatori g ON g.id = a.giocatore_id
        JOIN {NS}.squadre s ON s.nome = a.squadra_acquirente
        WHERE {where_clause}
        ORDER BY g.ruolo, g.nome
    """, params)


def esegui_strategia_asta():
    presi_rows = query_rows(f"""
        SELECT g.ruolo, COUNT(*) AS presi
        FROM {NS}.mia_rosa mr
        JOIN {NS}.giocatori g ON g.id = mr.giocatore_id
        GROUP BY g.ruolo
    """)
    presi_per_ruolo = {r["ruolo"]: r["presi"] for r in presi_rows}

    budget_rows = query_rows(f"""
        SELECT s.budget_totale,
               COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
        FROM {NS}.squadre s
        LEFT JOIN {NS}.asta_log a ON a.squadra_acquirente = s.nome
        WHERE lower(s.nome) = lower('{MY_TEAM}')
        GROUP BY s.budget_totale
    """)
    if not budget_rows:
        return {"errore": f"Squadra '{MY_TEAM}' non trovata in squadre. Verifica il censimento."}
    row = budget_rows[0]
    budget_residuo = float(row["budget_totale"]) - float(row["budget_speso"])

    mercato_rows = query_rows(f"""
        SELECT g.ruolo,
               AVG(a.prezzo_finale) AS prezzo_medio,
               AVG(g.fvm)           AS fvm_medio,
               COUNT(*)             AS n_acquisti
        FROM {NS}.asta_log a
        JOIN {NS}.giocatori g ON g.id = a.giocatore_id
        GROUP BY g.ruolo
    """)
    mercato = {}
    for r in mercato_rows:
        fvm_medio    = float(r["fvm_medio"] or 0)
        prezzo_medio = float(r["prezzo_medio"] or 0)
        rapporto     = (prezzo_medio / fvm_medio) if fvm_medio > 0 else None
        if rapporto is None:
            valutazione = "dati insufficienti"
        elif rapporto > 1.15:
            valutazione = "gonfiato (si paga sopra FVM)"
        elif rapporto < 0.85:
            valutazione = "conveniente (si paga sotto FVM)"
        else:
            valutazione = "in linea con FVM"
        mercato[r["ruolo"]] = {
            "prezzo_medio_pagato": round(prezzo_medio, 1),
            "fvm_medio":           round(fvm_medio, 1),
            "valutazione":         valutazione,
            "campione":            r["n_acquisti"],
        }

    slot_mancanti = {
        ruolo: max(ROSA_TARGET[ruolo] - presi_per_ruolo.get(ruolo, 0), 0)
        for ruolo in ROSA_TARGET
    }
    totale_slot = sum(slot_mancanti.values())
    budget_medio_per_slot = round(budget_residuo / totale_slot, 1) if totale_slot else 0

    candidati_per_ruolo = {}
    for ruolo, mancanti in slot_mancanti.items():
        if mancanti == 0:
            continue
        candidati_per_ruolo[ruolo] = query_rows(f"""
            SELECT nome, squadra, quotazione_attuale, fvm
            FROM {NS}.giocatori
            WHERE ruolo = ?
              AND id NOT IN (SELECT giocatore_id FROM {NS}.asta_log WHERE giocatore_id IS NOT NULL)
            ORDER BY fvm DESC NULLS LAST
            LIMIT 5
        """, [ruolo])

    return {
        "budget_residuo": budget_residuo,
        "budget_medio_per_slot_rimanente": budget_medio_per_slot,
        "rosa": {
            ruolo: {
                "richiesti": ROSA_TARGET[ruolo],
                "presi":     presi_per_ruolo.get(ruolo, 0),
                "mancanti":  slot_mancanti[ruolo],
            }
            for ruolo in ROSA_TARGET
        },
        "andamento_mercato_per_ruolo": mercato,
        "top_candidati_per_ruolo": candidati_per_ruolo,
    }


def get_stagione_corrente() -> str | None:
    rows = query_rows(f"""
        SELECT stagione FROM {NS}.statistiche_storiche
        ORDER BY stagione DESC LIMIT 1
    """)
    return rows[0]["stagione"] if rows else None

# ---------------------------------------------------------------------------
# Tool definitions (identiche all'originale — JSON puro, nessuna modifica)
# ---------------------------------------------------------------------------
TOOLS_BASE = [
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
                    "ruolo":          {"type": "string", "enum": ["P", "D", "C", "A"]},
                    "squadra":        {"type": "string"},
                    "escludi_squadra":{"type": "string"},
                    "order_by":       {"type": "string", "enum": list(COLONNE_ORDINABILI.keys())},
                    "ordine":         {"type": "string", "enum": ["asc", "desc"]},
                    "limit":          {"type": "integer"},
                    "solo_liberi":    {"type": "boolean"},
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
                "specifico nominato dall'utente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string"},
                    "top_k":  {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_BUDGET = {
    "type": "function",
    "function": {
        "name": "stato_mio_budget",
        "description": (
            "Ritorna il budget totale, speso e residuo della MIA squadra "
            "in questo momento dell'asta."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_ROSA = {
    "type": "function",
    "function": {
        "name": "la_mia_rosa",
        "description": (
            "Ritorna l'elenco ESATTO dei giocatori attualmente nella MIA rosa. "
            "Usa SEMPRE questo strumento per domande tipo 'chi ho in rosa'."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_ROSA_ALTRUI = {
    "type": "function",
    "function": {
        "name": "rosa_di_squadra",
        "description": (
            "Ritorna l'elenco ESATTO dei giocatori acquistati da una squadra "
            "specifica della lega (non la mia). Usa per QUALSIASI domanda "
            "sulla rosa di un'altra squadra/persona."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "nome_squadra": {"type": "string"},
            },
            "required": ["nome_squadra"],
        },
    },
}

TOOL_STRATEGIA = {
    "type": "function",
    "function": {
        "name": "strategia_asta",
        "description": (
            "Fotografia completa della situazione in asta: slot mancanti per "
            "ruolo, budget residuo, budget medio per slot, candidati liberi, "
            "andamento mercato. Usa per domande di strategia generale."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_STATISTICHE_STORICHE = {
    "type": "function",
    "function": {
        "name": "cerca_statistiche_storiche",
        "description": (
            "Interroga le statistiche storiche stagionali dei giocatori "
            "(presenze, media voto, fantamedia, gol, assist, ammonizioni, ecc.). "
            "Usa per domande tipo 'chi ha segnato di piu' negli ultimi anni', "
            "'top 10 per fantamedia nella stagione 2023/24', 'storico di Vlahovic'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "stagione":  {"type": "string"},
                "ruolo":     {"type": "string", "enum": ["P", "D", "C", "A"]},
                "nome":      {"type": "string"},
                "order_by":  {"type": "string", "enum": list(COLONNE_STAT_ORDINABILI.keys())},
                "ordine":    {"type": "string", "enum": ["asc", "desc"]},
                "limit":     {"type": "integer"},
            },
            "required": ["order_by", "ordine"],
        },
    },
}

TOOLS_GENERALE = TOOLS_BASE + [TOOL_ROSA, TOOL_ROSA_ALTRUI, TOOL_STATISTICHE_STORICHE]
TOOLS_ASTA     = TOOLS_BASE + [TOOL_BUDGET, TOOL_ROSA, TOOL_ROSA_ALTRUI, TOOL_STRATEGIA, TOOL_STATISTICHE_STORICHE]

SYSTEM_PROMPT_ASTA = (
    "Sei un assistente esperto di Fantacalcio, in asta live. Hai sette "
    "strumenti: uno per query strutturate su giocatori (ordinamenti/"
    "filtri/top-N, di default solo tra i liberi), uno per ricerca "
    "semantica (domande aperte), uno per sapere il MIO budget "
    "residuo, uno per sapere la MIA rosa attuale, uno per la rosa di "
    "ALTRE squadre della lega, uno per una fotografia completa della mia "
    "situazione in asta (slot mancanti per ruolo, budget medio per slot, "
    "candidati liberi, andamento mercato), uno per le statistiche storiche "
    "stagionali (presenze, gol, assist, fantamedia degli anni passati). "
    "Usa cerca_statistiche_storiche per domande sulle stagioni precedenti "
    "('quanto ha segnato X negli ultimi anni', 'top per fantamedia nel "
    "2023/24', 'storico di Y') — queste info aiutano a valutare un "
    "giocatore durante l'asta. Quando l'utente chiede "
    "consigli su chi prendere per un singolo acquisto, controlla SEMPRE "
    "prima il budget residuo con lo strumento dedicato, poi cerca "
    "giocatori liberi compatibili con quel budget. Per domande di "
    "strategia generale ('come sono messo', 'cosa mi manca', 'su chi "
    "punto ora') usa strategia_asta, che da' gia' tutto il quadro in una "
    "sola chiamata - non serve comporlo con altri strumenti. Per 'chi ha "
    "il valore piu alto/basso' o 'top N' usa SEMPRE lo strumento SQL, mai "
    "la ricerca semantica. Per la mia rosa/squadra usa SEMPRE lo "
    "strumento la_mia_rosa; per QUALSIASI domanda sulla rosa o acquisti "
    "di una squadra/persona diversa da me usa SEMPRE rosa_di_squadra, "
    "mai la_mia_rosa ne' la ricerca semantica. Quando rispondi su "
    "rosa_di_squadra, usa SEMPRE il nome squadra vero e l'allenatore "
    "restituiti dallo strumento. Rispondi SOLO in base ai risultati degli "
    "strumenti, mai inventando dati. "
    "Non annunciare MAI un'azione futura senza eseguirla nella stessa risposta."
)

SYSTEM_PROMPT_GENERALE = (
    "Sei un assistente esperto di Fantacalcio, per la gestione della "
    "squadra durante il campionato (formazioni, statistiche, chi "
    "schierare, confronti tra giocatori). Hai cinque strumenti: uno per "
    "query strutturate (ordinamenti/filtri/top-N su dati esatti), uno "
    "per ricerca semantica (domande aperte o su un giocatore specifico), "
    "uno per sapere la MIA rosa attuale, uno per la rosa di ALTRE "
    "squadre della lega, uno per le statistiche storiche stagionali. "
    "Usa cerca_statistiche_storiche per domande sulle stagioni precedenti. "
    "Per 'chi ha il valore piu alto/basso' o 'top N' usa SEMPRE lo "
    "strumento SQL. Per la mia rosa usa SEMPRE la_mia_rosa; per squadre "
    "altrui usa SEMPRE rosa_di_squadra. Rispondi SOLO in base ai "
    "risultati degli strumenti, mai inventando dati. "
    "Non annunciare MAI un'azione futura senza eseguirla nella stessa risposta."
)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="FantAssistant Chatbot — Databricks")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request with timing."""
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    _log_entry(
        "INFO",
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
        extra={"method": request.method, "path": str(request.url.path), "status": response.status_code, "ms": duration_ms},
    )
    return response


@app.get("/app-logs")
def get_app_logs(n: int = 100):
    """Return the last N log entries from the in-memory buffer."""
    entries = list(LOG_BUFFER)[-min(n, 500):]
    return {"count": len(entries), "logs": entries}


class ChatRequest(BaseModel):
    domanda: str
    top_k: int = 8
    modalita: str = "asta"
    storico: list[dict] = []


class ChatResponse(BaseModel):
    risposta: str
    contesto_usato: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "backend": "databricks"}


@app.post("/ingest")
def ingest():
    """
    Su Databricks non serve ricostruire manualmente la collection:
    il Vector Search Delta Sync si aggiorna automaticamente quando
    la tabella giocatori cambia (pipeline impostata nel notebook 02).
    Mantenuto per compatibilita' API con il frontend.
    """
    return {
        "status": "ok",
        "message": "Vector Search sync gestito automaticamente da Databricks Delta Sync.",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    _log_entry("INFO", f"Chat request: modalita={req.modalita}", extra={"domanda": req.domanda[:200]})
    tools         = TOOLS_ASTA     if req.modalita == "asta" else TOOLS_GENERALE
    system_prompt = SYSTEM_PROMPT_ASTA if req.modalita == "asta" else SYSTEM_PROMPT_GENERALE

    stagione_corrente = get_stagione_corrente()
    if stagione_corrente:
        system_prompt += (
            f" La stagione piu' recente disponibile nel database e' {stagione_corrente}: "
            "quando l'utente dice 'ultima stagione', 'quest anno' o 'stagione corrente' "
            "intende sempre questa."
        )

    messages = (
        [{"role": "system", "content": system_prompt}]
        + req.storico
        + [{"role": "user", "content": req.domanda}]
    )
    contesto_usato: list[str] = []

    resp = client_ai.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            name = tool_call.function.name

            if name == "cerca_giocatori_sql":
                righe = esegui_cerca_giocatori_sql(**args)
                risultato = righe
                contesto_usato.extend(
                    f"{r['nome']} ({r['ruolo']}, {r['squadra']}): "
                    f"quot. {r['quotazione_attuale']}, FVM {r['fvm']}"
                    for r in righe
                )
            elif name == "ricerca_semantica":
                risultato = esegui_ricerca_semantica(
                    args.get("query", req.domanda), args.get("top_k", req.top_k)
                )
                contesto_usato.extend(risultato)
            elif name == "stato_mio_budget":
                risultato = esegui_mio_budget()
                contesto_usato.append(
                    f"Mio budget: {risultato.get('budget_residuo')} residui su {risultato.get('budget_totale')}"
                )
            elif name == "la_mia_rosa":
                righe = esegui_mia_rosa()
                risultato = righe
                contesto_usato.extend(
                    f"{r['nome']} ({r['ruolo']}, {r['squadra']}), pagato {r['prezzo_pagato']}"
                    for r in righe
                ) if righe else contesto_usato.append("Nessun giocatore ancora in rosa.")
            elif name == "rosa_di_squadra":
                righe = esegui_rosa_di_squadra(**args)
                risultato = righe
                if righe:
                    contesto_usato.append(
                        f"Squadra trovata: '{righe[0]['nome_squadra']}' "
                        f"(allenatore: {righe[0]['allenatore']})"
                    )
                    contesto_usato.extend(
                        f"{r['nome']} ({r['ruolo']}, {r['squadra']}), pagato {r['prezzo_finale']}"
                        for r in righe
                    )
                else:
                    contesto_usato.append(f"Nessuna squadra trovata per '{args.get('nome_squadra')}'.")
            elif name == "strategia_asta":
                risultato = esegui_strategia_asta()
                contesto_usato.append(json.dumps(risultato, default=str, ensure_ascii=False))
            elif name == "cerca_statistiche_storiche":
                righe = esegui_statistiche_storiche(**args)
                risultato = righe
                contesto_usato.extend(
                    f"{r['nome']} ({r['ruolo']}, {r['squadra']}) "
                    f"stagione {r['stagione']}: "
                    f"fantamedia {r['fantamedia']}, gol {r['gol']}, "
                    f"assist {r['assist']}, presenze {r['presenze']}"
                    for r in righe
                )
            else:
                risultato = []

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      json.dumps(risultato, default=str, ensure_ascii=False),
            })

        resp_finale = client_ai.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0.3,
        )
        risposta = resp_finale.choices[0].message.content
    else:
        risposta = msg.content

    _log_entry("INFO", f"Chat completed: {len(contesto_usato)} context items")
    return ChatResponse(risposta=risposta, contesto_usato=contesto_usato)


@app.on_event("startup")
def on_startup():
    _log_entry("INFO", "FantAssistant chatbot started")
    _log_entry("INFO", f"Log table: {LOG_TABLE} (pre-created externally)")

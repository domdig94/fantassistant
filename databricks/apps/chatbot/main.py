"""
FantAssistant Databricks — Chatbot App con AI Gateway nativo Databricks

Variante di main.py che usa gli endpoint LLM esposti direttamente da
Databricks (Model Serving / AI Gateway) invece di Azure OpenAI.

Vantaggi rispetto a main.py:
  - Nessuna dipendenza da Azure OpenAI endpoint esterno
  - Il token di autenticazione e' lo stesso Databricks (DATABRICKS_TOKEN),
    gia' disponibile nel runtime App — zero secrets aggiuntivi
  - Puoi switchare modello cambiando solo DBX_LLM_ENDPOINT env var
  - Logging/usage tracking centralizzato nell'AI Gateway ISP

Come funziona:
  Il client openai viene puntato al serving endpoint Databricks,
  che espone un'API identica a OpenAI (stesso schema request/response).
  Il function calling funziona esattamente uguale.

Endpoint disponibili (scegli con DBX_LLM_ENDPOINT):
  - databricks-meta-llama-3-3-70b-instruct   (veloce, function calling ok)
  - databricks-claude-sonnet-4               (piu' capace, piu' lento)
  - databricks-claude-opus-4-8               (consigliato: massima qualita')
  - databricks-gpt-oss-120b                  (alternativa open)
  - seps-mer10-aiaastest-gpt-5-4-2026-03-05  (GPT-5.4 via AI Gateway ISP)

Env vars richieste (tutte auto-disponibili nel runtime App tranne DBX_LLM_ENDPOINT):
  DATABRICKS_HOST         auto-iniettato dal runtime
  DATABRICKS_TOKEN        auto-iniettato dal runtime
  DATABRICKS_SQL_HTTP_PATH  da secret scope
  DBX_LLM_ENDPOINT        nome endpoint LLM (default: databricks-claude-opus-4-8)
  UNITY_CATALOG           default: fantassistant
  UNITY_SCHEMA            default: main
  MY_TEAM                 da secret scope
  BUDGET_LEGA             da secret scope
  VECTOR_SEARCH_ENDPOINT  da secret scope
"""
import os
import json
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.sql import StatementState, StatementParameterListItem

_cfg = Config()  # usa automaticamente l'auth del service principal dell'app
_ws = WorkspaceClient(config=_cfg)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _get_databricks_token() -> str:
    """Token da env var o SDK Config (supporta PAT e OAuth service principal)."""
    token = _cfg.token
    if token:
        return token
    # OAuth SP: il token non e' statico, lo estraiamo dall'header factory
    headers = _cfg.authenticate()
    return headers.get("Authorization", "").removeprefix("Bearer ")


DATABRICKS_HOST      = _cfg.host
DATABRICKS_TOKEN     = _get_databricks_token() 
DATABRICKS_HTTP_PATH = os.environ["DATABRICKS_SQL_HTTP_PATH"]
CATALOG  = os.environ.get("UNITY_CATALOG", "platform")
SCHEMA   = os.environ.get("UNITY_SCHEMA",  "fantassistant")
NS       = f"`{CATALOG}`.`{SCHEMA}`"

VS_ENDPOINT = os.environ["VECTOR_SEARCH_ENDPOINT"]
VS_INDEX    = f"{CATALOG}.{SCHEMA}.giocatori_vs_index"

MY_TEAM     = os.environ.get("MY_TEAM", "Io")
BUDGET_LEGA = int(os.environ.get("BUDGET_LEGA", "1000"))
ROSA_TARGET = {
    "P": int(os.environ.get("ROSA_TARGET_P", 3)),
    "D": int(os.environ.get("ROSA_TARGET_D", 8)),
    "C": int(os.environ.get("ROSA_TARGET_C", 8)),
    "A": int(os.environ.get("ROSA_TARGET_A", 6)),
}

# ---------------------------------------------------------------------------
# Client LLM — punta al serving endpoint Databricks (API OpenAI-compatible)
# ---------------------------------------------------------------------------
DBX_LLM_ENDPOINT = os.environ.get(
    "DBX_LLM_ENDPOINT",
    "databricks-claude-opus-4-8",
)

client_ai = OpenAI(
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
    api_key=DATABRICKS_TOKEN,
)

# Con l'AI Gateway il campo "model" corrisponde al nome endpoint;
# la routing e' gestita dall'AI Gateway.
CHAT_MODEL = DBX_LLM_ENDPOINT

# ---------------------------------------------------------------------------
# DB helpers — Statement Execution API (REST, bypassa Thrift)
# ---------------------------------------------------------------------------
_WAREHOUSE_ID = DATABRICKS_HTTP_PATH.rstrip("/").split("/")[-1]


def _cast(value: str | None, type_name):
    """Converte il valore stringa di data_array nel tipo Python corretto."""
    if value is None:
        return None
    t = (type_name.value if hasattr(type_name, 'value') else str(type_name or "")).upper()
    if t in ("INT", "BIGINT", "SMALLINT", "TINYINT", "LONG"):
        return int(value)
    if t in ("DOUBLE", "FLOAT") or t.startswith("DECIMAL"):
        return float(value)
    if t == "BOOLEAN":
        return value.lower() == "true"
    return value


def query_rows(sql: str, params: dict[str, str] | None = None) -> list[dict]:
    stmt_params = None
    if params:
        stmt_params = [
            StatementParameterListItem(name=k, value=str(v))
            for k, v in params.items()
        ]
    response = _ws.statement_execution.execute_statement(
        warehouse_id=_WAREHOUSE_ID,
        statement=sql,
        parameters=stmt_params,
        wait_timeout="30s",
    )
    if response.status.state != StatementState.SUCCEEDED:
        err = response.status.error
        raise RuntimeError(f"SQL failed ({response.status.state}): {err}")
    columns = response.manifest.schema.columns
    cols = [c.name for c in columns]
    types = [c.type_name for c in columns]
    rows = response.result.data_array or []
    return [
        {col: _cast(val, typ) for col, val, typ in zip(cols, row, types)}
        for row in rows
    ]

# ---------------------------------------------------------------------------
# Vector Search
# ---------------------------------------------------------------------------
_vs_client = None

def get_vs_client() -> VectorSearchClient:
    global _vs_client
    if _vs_client is None:
        _vs_client = VectorSearchClient(
            workspace_url=DATABRICKS_HOST,
            personal_access_token=DATABRICKS_TOKEN,
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
    cols     = [c["name"] for c in results.get("result", {}).get("columns", [])]
    rows     = results.get("result", {}).get("data_array", [])
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
# Tool implementations (identiche a main.py)
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
    params = {}
    if ruolo:
        sql += " AND ruolo = :ruolo"
        params["ruolo"] = ruolo
    if squadra:
        sql += " AND lower(squadra) LIKE lower(:squadra_pattern)"
        params["squadra_pattern"] = f"%{squadra}%"
    if escludi_squadra:
        sql += " AND lower(squadra) NOT LIKE lower(:escludi_squadra_pattern)"
        params["escludi_squadra_pattern"] = f"%{escludi_squadra}%"
    if solo_liberi:
        sql += f" AND id NOT IN (SELECT giocatore_id FROM {NS}.asta_log WHERE giocatore_id IS NOT NULL)"
    sql += f" ORDER BY {colonna} {direzione} NULLS LAST LIMIT {limit}"
    return query_rows(sql, params or None)


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
    params = {}
    if stagione:
        sql += " AND s.stagione = :stagione"
        params["stagione"] = stagione
    if ruolo:
        sql += " AND g.ruolo = :ruolo"
        params["ruolo"] = ruolo
    if nome:
        sql += " AND lower(g.nome) LIKE lower(:nome_pattern)"
        params["nome_pattern"] = f"%{nome}%"
    sql += f" ORDER BY {colonna} {direzione} NULLS LAST LIMIT {limit}"
    return query_rows(sql, params or None)


def esegui_mio_budget():
    rows = query_rows(f"""
        SELECT s.nome, s.budget_totale,
               COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
        FROM {NS}.squadre s
        LEFT JOIN {NS}.asta_log a ON a.squadra_acquirente = s.nome
        WHERE lower(s.nome) = lower(:my_team)
        GROUP BY s.nome, s.budget_totale
    """, {"my_team": MY_TEAM})
    if not rows:
        return {"errore": f"Squadra '{MY_TEAM}' non trovata. Verifica il censimento."}
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
    condizioni, params = [], {}
    for i, p in enumerate(parole):
        nome_key = f"nome_{i}"
        allen_key = f"allen_{i}"
        condizioni.append(f"(lower(s.nome) LIKE lower(:{nome_key}) OR lower(s.allenatore) LIKE lower(:{allen_key}))")
        params[nome_key] = f"%{p}%"
        params[allen_key] = f"%{p}%"
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
        WHERE lower(s.nome) = lower(:my_team)
        GROUP BY s.budget_totale
    """, {"my_team": MY_TEAM})
    if not budget_rows:
        return {"errore": f"Squadra '{MY_TEAM}' non trovata. Verifica il censimento."}
    row           = budget_rows[0]
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
    totale_slot        = sum(slot_mancanti.values())
    budget_medio_slot  = round(budget_residuo / totale_slot, 1) if totale_slot else 0

    candidati_per_ruolo = {}
    for ruolo, mancanti in slot_mancanti.items():
        if mancanti == 0:
            continue
        candidati_per_ruolo[ruolo] = query_rows(f"""
            SELECT nome, squadra, quotazione_attuale, fvm
            FROM {NS}.giocatori
            WHERE ruolo = :ruolo
              AND id NOT IN (SELECT giocatore_id FROM {NS}.asta_log WHERE giocatore_id IS NOT NULL)
            ORDER BY fvm DESC NULLS LAST
            LIMIT 5
        """, {"ruolo": ruolo})

    return {
        "budget_residuo":                budget_residuo,
        "budget_medio_per_slot_rimanente": budget_medio_slot,
        "rosa": {
            ruolo: {
                "richiesti": ROSA_TARGET[ruolo],
                "presi":     presi_per_ruolo.get(ruolo, 0),
                "mancanti":  slot_mancanti[ruolo],
            }
            for ruolo in ROSA_TARGET
        },
        "andamento_mercato_per_ruolo": mercato,
        "top_candidati_per_ruolo":     candidati_per_ruolo,
    }


def get_stagione_corrente() -> str | None:
    rows = query_rows(f"""
        SELECT stagione FROM {NS}.statistiche_storiche
        ORDER BY stagione DESC LIMIT 1
    """)
    return rows[0]["stagione"] if rows else None

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
TOOLS_BASE = [
    {
        "type": "function",
        "function": {
            "name": "cerca_giocatori_sql",
            "description": (
                "Cerca giocatori con filtri esatti e/o ordinamento su un "
                "campo numerico (FVM, quotazione). Usa per domande tipo "
                "'top N per X', 'chi ha il FVM piu alto', 'difensori sotto "
                "i 10 crediti'. Non usarlo per domande generiche."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ruolo":           {"type": "string", "enum": ["P", "D", "C", "A"]},
                    "squadra":         {"type": "string"},
                    "escludi_squadra": {"type": "string"},
                    "order_by":        {"type": "string", "enum": list(COLONNE_ORDINABILI.keys())},
                    "ordine":          {"type": "string", "enum": ["asc", "desc"]},
                    "limit":           {"type": "integer"},
                    "solo_liberi":     {"type": "boolean"},
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
                "Usa per domande aperte o su un giocatore specifico nominato."
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
        "description": "Ritorna budget totale, speso e residuo della MIA squadra.",
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_ROSA = {
    "type": "function",
    "function": {
        "name": "la_mia_rosa",
        "description": (
            "Ritorna l'elenco ESATTO dei giocatori nella MIA rosa. "
            "Usa SEMPRE per domande tipo 'chi ho in rosa'."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_ROSA_ALTRUI = {
    "type": "function",
    "function": {
        "name": "rosa_di_squadra",
        "description": (
            "Ritorna l'elenco ESATTO dei giocatori di una squadra altrui. "
            "Usa per QUALSIASI domanda sulla rosa di un'altra squadra/persona."
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
            "Fotografia completa della situazione in asta: slot mancanti, "
            "budget residuo, budget medio per slot, candidati liberi, "
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
            "Statistiche storiche stagionali (presenze, media voto, fantamedia, "
            "gol, assist, ammonizioni, ecc.). Usa per domande sulle stagioni "
            "precedenti o lo storico di un giocatore."
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
    "Usa cerca_statistiche_storiche per domande sulle stagioni precedenti. "
    "Quando l'utente chiede consigli su chi prendere, controlla SEMPRE "
    "prima il budget residuo, poi cerca giocatori liberi compatibili. "
    "Per domande di strategia generale usa strategia_asta. "
    "Per 'top N' o ordinamenti usa SEMPRE lo strumento SQL. "
    "Per la mia rosa usa SEMPRE la_mia_rosa; per squadre altrui usa "
    "SEMPRE rosa_di_squadra. Rispondi SOLO in base ai risultati degli "
    "strumenti, mai inventando dati. "
    "Non annunciare MAI un'azione futura senza eseguirla nella stessa risposta."
)

SYSTEM_PROMPT_GENERALE = (
    "Sei un assistente esperto di Fantacalcio per la gestione della squadra "
    "durante il campionato (formazioni, statistiche, chi schierare, confronti). "
    "Hai cinque strumenti: query strutturate, ricerca semantica, mia rosa, "
    "rosa altrui, statistiche storiche. "
    "Per 'top N' o ordinamenti usa SEMPRE lo strumento SQL. "
    "Per la mia rosa usa SEMPRE la_mia_rosa; per squadre altrui usa "
    "SEMPRE rosa_di_squadra. Rispondi SOLO in base ai risultati degli "
    "strumenti, mai inventando dati. "
    "Non annunciare MAI un'azione futura senza eseguirla nella stessa risposta."
)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="FantAssistant Chatbot — Databricks AI Gateway")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    domanda: str
    top_k:    int  = 8
    modalita: str  = "asta"
    storico:  list[dict] = []


class ChatResponse(BaseModel):
    risposta:       str
    contesto_usato: list[str]
    modello_usato:  str


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "backend":        "databricks-ai-gateway",
        "llm_endpoint":   DBX_LLM_ENDPOINT,
    }


@app.get("/check-sql")
def check_sql():
    """Diagnostica connessione SQL via Statement Execution API."""
    info = {
        "warehouse_id":  _WAREHOUSE_ID,
        "host_raw":      DATABRICKS_HOST,
        "has_token":     bool(_get_databricks_token()),
        "token_prefix":  _get_databricks_token()[:8] + "..." if _get_databricks_token() else None,
    }
    try:
        rows = query_rows("SELECT 1 AS test")
        info["sql_status"] = "ok"
        info["result"] = rows[0]["test"] if rows else None
    except Exception as e:
        info["sql_status"] = "error"
        info["error_type"] = type(e).__name__
        info["error_detail"] = str(e)[:500]
    return info


@app.post("/ingest")
def ingest():
    return {
        "status":  "ok",
        "message": "Vector Search sync gestito automaticamente da Databricks Delta Sync.",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    tools         = TOOLS_ASTA      if req.modalita == "asta" else TOOLS_GENERALE
    system_prompt = SYSTEM_PROMPT_ASTA if req.modalita == "asta" else SYSTEM_PROMPT_GENERALE

    stagione_corrente = get_stagione_corrente()
    if stagione_corrente:
        system_prompt += (
            f" La stagione piu' recente disponibile nel database e' {stagione_corrente}: "
            "quando l'utente dice 'ultima stagione' o 'stagione corrente' intende questa."
        )

    messages = (
        [{"role": "system", "content": system_prompt}]
        + req.storico
        + [{"role": "user", "content": req.domanda}]
    )
    contesto_usato: list[str] = []

    resp = client_ai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            name = tool_call.function.name

            if name == "cerca_giocatori_sql":
                righe     = esegui_cerca_giocatori_sql(**args)
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
                    f"Budget residuo: {risultato.get('budget_residuo')} "
                    f"su {risultato.get('budget_totale')}"
                )
            elif name == "la_mia_rosa":
                righe     = esegui_mia_rosa()
                risultato = righe
                if righe:
                    contesto_usato.extend(
                        f"{r['nome']} ({r['ruolo']}, {r['squadra']}), pagato {r['prezzo_pagato']}"
                        for r in righe
                    )
                else:
                    contesto_usato.append("Nessun giocatore ancora in rosa.")
            elif name == "rosa_di_squadra":
                righe     = esegui_rosa_di_squadra(**args)
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
                righe     = esegui_statistiche_storiche(**args)
                risultato = righe
                contesto_usato.extend(
                    f"{r['nome']} ({r['ruolo']}, {r['squadra']}) "
                    f"stagione {r['stagione']}: fantamedia {r['fantamedia']}, "
                    f"gol {r['gol']}, assist {r['assist']}, presenze {r['presenze']}"
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
            model=CHAT_MODEL,
            messages=messages,
        )
        risposta = resp_finale.choices[0].message.content
    else:
        risposta = msg.content

    return ChatResponse(
        risposta=risposta,
        contesto_usato=contesto_usato,
        modello_usato=DBX_LLM_ENDPOINT,
    )

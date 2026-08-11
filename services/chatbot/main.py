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
from fastapi.middleware.cors import CORSMiddleware
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
MY_TEAM = os.environ.get("MY_TEAM", "Io")

ROSA_TARGET = {
    "P": int(os.environ.get("ROSA_TARGET_P", 3)),
    "D": int(os.environ.get("ROSA_TARGET_D", 8)),
    "C": int(os.environ.get("ROSA_TARGET_C", 8)),
    "A": int(os.environ.get("ROSA_TARGET_A", 6)),
}


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


def esegui_statistiche_storiche(
    stagione=None,
    ruolo=None,
    order_by="fantamedia",
    ordine="desc",
    limit=10,
    nome=None,
):
    """
    Interroga statistiche_storiche con join su giocatori.
    Usata dal tool cerca_statistiche_storiche.
    """
    colonna = COLONNE_STAT_ORDINABILI.get(order_by, "fantamedia")
    direzione = "ASC" if ordine == "asc" else "DESC"
    limit = min(int(limit or 10), 50)

    query = """
        SELECT g.nome, g.ruolo, g.squadra,
               s.stagione, s.presenze, s.media_voto, s.fantamedia,
               s.gol, s.assist, s.ammonizioni, s.espulsioni,
               s.gol_subiti, s.rigori_parati, s.autogol
        FROM statistiche_storiche s
        JOIN giocatori g ON g.id = s.giocatore_id
        WHERE 1=1
    """
    params = []
    if stagione:
        query += " AND s.stagione = %s"
        params.append(stagione)
    if ruolo:
        query += " AND g.ruolo = %s"
        params.append(ruolo)
    if nome:
        query += " AND g.nome ILIKE %s"
        params.append(f"%{nome}%")
    query += f" ORDER BY {colonna} {direzione} NULLS LAST LIMIT %s"
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


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
                    "ruolo": {"type": "string", "enum": ["P", "D", "C", "A"], "description": "Filtra per ruolo. Omettere per tutti i ruoli."},
                    "squadra": {"type": "string", "description": "Filtra per squadra (match parziale). Omettere per tutte."},
                    "escludi_squadra": {"type": "string","description": "Escludi questa squadra dai risultati."},
                    "order_by": {"type": "string", "enum": list(COLONNE_ORDINABILI.keys()), "description": "Campo su cui ordinare."},
                    "ordine": {"type": "string", "enum": ["asc", "desc"], "description": "asc = dal piu basso, desc = dal piu alto."},
                    "limit": {"type": "integer", "description": "Numero massimo di risultati (default 10, max 50)."},
                    "solo_liberi": {"type": "boolean", "description": "Se true (default), esclude i giocatori gia' assegnati in asta."},
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

TOOL_BUDGET = {
    "type": "function",
    "function": {
        "name": "stato_mio_budget",
        "description": (
            "Ritorna il budget totale, speso e residuo della MIA squadra "
            "in questo momento dell'asta. Usa questo strumento ogni volta "
            "che l'utente chiede consigli su cosa conviene prendere, per "
            "sapere quanto puo' davvero spendere."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_ROSA = {
    "type": "function",
    "function": {
        "name": "la_mia_rosa",
        "description": (
            "Ritorna l'elenco ESATTO dei giocatori attualmente nella MIA "
            "rosa, presi dal database (non da similarita' testuale). Usa "
            "SEMPRE questo strumento per domande tipo 'chi ho in rosa', "
            "'quale e' la mia squadra', 'cosa ho preso finora'. Mai la "
            "ricerca semantica per questo tipo di domanda."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_ROSA_ALTRUI = {
    "type": "function",
    "function": {
        "name": "rosa_di_squadra",
        "description": (
            "Ritorna l'elenco ESATTO dei giocatori acquistati da una "
            "squadra specifica della lega (non la mia), preso da "
            "asta_log che contiene tutti gli acquisti di tutti. Usa "
            "questo strumento per QUALSIASI domanda che chiede la rosa "
            "o gli acquisti di una squadra/persona diversa da me: 'chi "
            "ha in rosa X', 'chi ha in squadra X', 'cosa ha preso X', "
            "'la squadra di X', dove X e' un nome di squadra o di "
            "allenatore. NON usare la_mia_rosa per questo (quella e' "
            "solo per la mia squadra)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "nome_squadra": {"type": "string", "description": "Nome della squadra o nome dell'allenatore di cui vedere gli acquisti."},
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
            "Ritorna una fotografia completa della mia situazione in asta: "
            "slot ancora da riempire per ruolo, budget residuo, budget medio "
            "disponibile per ogni slot ancora da riempire, i migliori "
            "candidati liberi per ogni ruolo mancante, e se il mercato sta "
            "pagando sopra o sotto l'FVM per ciascun ruolo. Usa questo "
            "strumento quando l'utente chiede una strategia generale, "
            "'come sono messo', 'cosa mi manca', 'su chi punto ora', o "
            "consigli complessivi (non su un singolo giocatore)."
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
            "(presenze, media voto, fantamedia, gol, assist, ammonizioni, "
            "espulsioni, gol subiti, rigori parati, autogol). Usa questo "
            "strumento per domande tipo 'chi ha segnato di più negli ultimi "
            "anni', 'top 10 per fantamedia nella stagione 2023/24', "
            "'storico di Vlahovic', 'chi è più affidabile negli ultimi 3 anni'. "
            "Puoi filtrare per stagione (es. '2024/25'), ruolo e nome giocatore, "
            "e ordinare per qualsiasi statistica."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "stagione": {
                    "type": "string",
                    "description": "Filtra per stagione, formato YYYY/YY (es. '2024/25'). Ometti per tutte le stagioni.",
                },
                "ruolo": {
                    "type": "string",
                    "enum": ["P", "D", "C", "A"],
                    "description": "Filtra per ruolo. Ometti per tutti.",
                },
                "nome": {
                    "type": "string",
                    "description": "Filtra per nome giocatore (match parziale). Utile per lo storico di un singolo.",
                },
                "order_by": {
                    "type": "string",
                    "enum": list(COLONNE_STAT_ORDINABILI.keys()),
                    "description": "Campo su cui ordinare i risultati.",
                },
                "ordine": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "asc = dal più basso, desc = dal più alto.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di risultati (default 10, max 50).",
                },
            },
            "required": ["order_by", "ordine"],
        },
    },
}

TOOLS_GENERALE = TOOLS_BASE + [TOOL_ROSA, TOOL_ROSA_ALTRUI, TOOL_STATISTICHE_STORICHE]
TOOLS_ASTA = TOOLS_BASE + [TOOL_BUDGET, TOOL_ROSA, TOOL_ROSA_ALTRUI, TOOL_STRATEGIA, TOOL_STATISTICHE_STORICHE]

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
    "di una squadra/persona diversa da me ('chi ha in rosa X', 'chi ha "
    "in squadra X', 'cosa ha preso X', 'la squadra di X', dove X puo' "
    "essere un nome di squadra O di allenatore) usa SEMPRE "
    "rosa_di_squadra, mai la_mia_rosa ne' la ricerca semantica - quella "
    "puo' restituire giocatori a caso. Quando rispondi su "
    "rosa_di_squadra, usa SEMPRE il nome squadra vero e l'allenatore "
    "restituiti dallo strumento, non ripetere il testo che ha scritto "
    "l'utente come se fosse il nome ufficiale. Rispondi SOLO in base ai "
    "risultati degli strumenti, mai inventando dati. Se i risultati non "
    "bastano, dillo. "
    "Non annunciare MAI un'azione futura (es. 'ora cerco', 'un attimo', 'ci "
    "penso') senza eseguirla nella stessa risposta: o chiami subito lo "
    "strumento giusto e rispondi con il risultato, oppure rispondi "
    "direttamente. Mai lasciare una richiesta a meta'."
)

SYSTEM_PROMPT_GENERALE = (
    "Sei un assistente esperto di Fantacalcio, per la gestione della "
    "squadra durante il campionato (formazioni, statistiche, chi "
    "schierare, confronti tra giocatori). Hai cinque strumenti: uno per "
    "query strutturate (ordinamenti/filtri/top-N su dati esatti), uno "
    "per ricerca semantica (domande aperte o su un giocatore specifico), "
    "uno per sapere la MIA rosa attuale, uno per la rosa di ALTRE "
    "squadre della lega, uno per le statistiche storiche stagionali "
    "(presenze, gol, assist, fantamedia degli anni passati). "
    "Usa cerca_statistiche_storiche per domande sulle stagioni precedenti "
    "('quanto ha segnato X negli ultimi anni', 'top per fantamedia', "
    "'storico di Y', 'chi schiero tra X e Y guardando la storia'). "
    "Per 'chi ha il valore piu alto/basso' o 'top N' "
    "usa SEMPRE lo strumento SQL, mai la ricerca semantica. Per la mia "
    "rosa/squadra usa SEMPRE lo strumento la_mia_rosa; per QUALSIASI "
    "domanda sulla rosa o acquisti di una squadra/persona diversa da me "
    "('chi ha in rosa X', 'chi ha in squadra X', 'cosa ha preso X', 'la "
    "squadra di X', dove X puo' essere un nome di squadra O di "
    "allenatore) usa SEMPRE rosa_di_squadra, mai la_mia_rosa ne' la "
    "ricerca semantica - quella puo' restituire giocatori a caso. "
    "Quando rispondi su rosa_di_squadra, usa SEMPRE il nome squadra "
    "vero e l'allenatore restituiti dallo strumento, non ripetere il "
    "testo che ha scritto l'utente come se fosse il nome ufficiale. "
    "Rispondi SOLO in base ai risultati degli strumenti, mai inventando "
    "dati. Se i risultati non bastano, dillo. "
    "Non annunciare MAI un'azione futura (es. 'ora cerco', 'un attimo', 'ci "
    "penso') senza eseguirla nella stessa risposta: o chiami subito lo "
    "strumento giusto e rispondi con il risultato, oppure rispondi "
    "direttamente. Mai lasciare una richiesta a meta'."
)


def esegui_cerca_giocatori_sql(ruolo=None, squadra=None, escludi_squadra=None, order_by="fvm", ordine="desc", limit=10, solo_liberi=True):
    colonna = COLONNE_ORDINABILI.get(order_by, "fvm")
    direzione = "ASC" if ordine == "asc" else "DESC"
    limit = min(int(limit or 10), 50)

    query = "SELECT nome, ruolo, squadra, quotazione_attuale, fvm FROM giocatori WHERE 1=1"
    params = []
    if ruolo:
        query += " AND ruolo = %s"
        params.append(ruolo)
    if squadra:
        query += " AND squadra ILIKE %s"
        params.append(f"%{squadra}%")
    if escludi_squadra:
        query += " AND squadra NOT ILIKE %s"
        params.append(f"%{escludi_squadra}%")
    if solo_liberi:
        query += " AND id NOT IN (SELECT giocatore_id FROM asta_log)"
    query += f" ORDER BY {colonna} {direzione} NULLS LAST LIMIT %s"
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def esegui_mio_budget():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.nome, s.budget_totale,
                       COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
                FROM squadre s
                LEFT JOIN asta_log a ON a.squadra_acquirente = s.nome
                WHERE LOWER(s.nome) = LOWER(%s)
                GROUP BY s.nome, s.budget_totale
                """,
                (MY_TEAM,),
            )
            r = cur.fetchone()
    if not r:
        return {
            "errore": (
                f"Squadra '{MY_TEAM}' (da MY_TEAM) non trovata in squadre. "
                f"Verifica il censimento con GET /squadre su auction-tracker."
            )
        }
    return {
        "budget_totale": float(r["budget_totale"]),
        "budget_speso": float(r["budget_speso"]),
        "budget_residuo": float(r["budget_totale"]) - float(r["budget_speso"]),
    }


def esegui_mia_rosa():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.nome, g.ruolo, g.squadra, mr.prezzo_pagato
                FROM mia_rosa mr
                JOIN giocatori g ON g.id = mr.giocatore_id
                ORDER BY g.ruolo, g.nome
                """
            )
            return cur.fetchall()
        

def esegui_rosa_di_squadra(nome_squadra: str):
    parole = [p for p in nome_squadra.strip().split() if p]
    if not parole:
        return []

    condizioni, params = [], []
    for p in parole:
        condizioni.append("(s.nome ILIKE %s OR s.allenatore ILIKE %s)")
        params.extend([f"%{p}%", f"%{p}%"])
    where_clause = " OR ".join(condizioni)

    query = f"""
        SELECT g.nome, g.ruolo, g.squadra, a.prezzo_finale,
               s.nome AS nome_squadra, s.allenatore
        FROM asta_log a
        JOIN giocatori g ON g.id = a.giocatore_id
        JOIN squadre s ON s.nome = a.squadra_acquirente
        WHERE {where_clause}
        ORDER BY g.ruolo, g.nome
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
        

def esegui_strategia_asta():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.ruolo, COUNT(*) AS presi
                FROM mia_rosa mr JOIN giocatori g ON g.id = mr.giocatore_id
                GROUP BY g.ruolo
                """
            )
            presi_per_ruolo = {r["ruolo"]: r["presi"] for r in cur.fetchall()}

            cur.execute(
                """
                SELECT s.budget_totale,
                       COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
                FROM squadre s
                LEFT JOIN asta_log a ON a.squadra_acquirente = s.nome
                WHERE LOWER(s.nome) = LOWER(%s)
                GROUP BY s.budget_totale
                """,
                (MY_TEAM,),
            )
            row = cur.fetchone()
            if not row:
                return {
                    "errore": (
                        f"Squadra '{MY_TEAM}' non trovata in squadre. "
                        f"Verifica il censimento."
                    )
                }
            budget_residuo = float(row["budget_totale"]) - float(row["budget_speso"])

            cur.execute(
                """
                SELECT g.ruolo, AVG(a.prezzo_finale) AS prezzo_medio,
                       AVG(g.fvm) AS fvm_medio, COUNT(*) AS n_acquisti
                FROM asta_log a JOIN giocatori g ON g.id = a.giocatore_id
                GROUP BY g.ruolo
                """
            )
            mercato = {}
            for r in cur.fetchall():
                fvm_medio = float(r["fvm_medio"] or 0)
                prezzo_medio = float(r["prezzo_medio"] or 0)
                rapporto = (prezzo_medio / fvm_medio) if fvm_medio > 0 else None
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
                    "fvm_medio": round(fvm_medio, 1),
                    "valutazione": valutazione,
                    "campione": r["n_acquisti"],
                }

            slot_mancanti = {
                ruolo: max(target - presi_per_ruolo.get(ruolo, 0), 0)
                for ruolo, target in ROSA_TARGET.items()
            }
            totale_slot_mancanti = sum(slot_mancanti.values())
            budget_medio_per_slot = (
                round(budget_residuo / totale_slot_mancanti, 1) if totale_slot_mancanti else 0
            )

            candidati_per_ruolo = {}
            for ruolo, mancanti in slot_mancanti.items():
                if mancanti == 0:
                    continue
                cur.execute(
                    """
                    SELECT nome, squadra, quotazione_attuale, fvm
                    FROM giocatori
                    WHERE ruolo = %s AND id NOT IN (SELECT giocatore_id FROM asta_log)
                    ORDER BY fvm DESC NULLS LAST
                    LIMIT 5
                    """,
                    (ruolo,),
                )
                candidati_per_ruolo[ruolo] = cur.fetchall()

    return {
        "budget_residuo": budget_residuo,
        "budget_medio_per_slot_rimanente": budget_medio_per_slot,
        "rosa": {
            ruolo: {
                "richiesti": ROSA_TARGET[ruolo],
                "presi": presi_per_ruolo.get(ruolo, 0),
                "mancanti": slot_mancanti[ruolo],
            }
            for ruolo in ROSA_TARGET
        },
        "andamento_mercato_per_ruolo": mercato,
        "top_candidati_per_ruolo": {
            ruolo: [dict(c) for c in candidati]
            for ruolo, candidati in candidati_per_ruolo.items()
        },
    }


def esegui_ricerca_semantica(query: str, top_k: int = 8):
    risultati = get_collection().query(query_texts=[query], n_results=top_k)
    return risultati["documents"][0] if risultati["documents"] else []


def get_stagione_corrente() -> str | None:
    """Ritorna la stagione più recente presente nel DB (es. '2025/26')."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT stagione FROM statistiche_storiche ORDER BY stagione DESC LIMIT 1"
            )
            r = cur.fetchone()
    return r["stagione"] if r else None


app = FastAPI(title="FantAssistant Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    domanda: str
    top_k: int = 8
    modalita: str = "asta"   # "asta" | "generale"
    storico: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


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
                       mr.prezzo_pagato,
                       array_agg(
                           json_build_object(
                               'stagione', s.stagione,
                               'presenze', s.presenze,
                               'gol',      s.gol,
                               'assist',   s.assist,
                               'media_voto', s.media_voto,
                               'fantamedia', s.fantamedia
                           ) ORDER BY s.stagione DESC
                       ) FILTER (WHERE s.stagione IS NOT NULL) AS stagioni
                FROM giocatori g
                LEFT JOIN statistiche_storiche s ON s.giocatore_id = g.id
                LEFT JOIN mia_rosa mr ON mr.giocatore_id = g.id
                GROUP BY g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale, g.fvm, mr.prezzo_pagato
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
        if r["stagioni"]:
            for s in r["stagioni"]:
                testo += (
                    f"Stagione {s['stagione']}: {s['presenze']} presenze, "
                    f"{s['gol']} gol, {s['assist']} assist, "
                    f"media voto {s['media_voto']}, "
                    f"fantamedia {s['fantamedia']}. "
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
    tools = TOOLS_ASTA if req.modalita == "asta" else TOOLS_GENERALE
    system_prompt = SYSTEM_PROMPT_ASTA if req.modalita == "asta" else SYSTEM_PROMPT_GENERALE

    # Arricchisce il system prompt con la stagione corrente ricavata dal DB,
    # così il modello sa cosa intende l'utente con "ultima stagione" o "quest'anno".
    stagione_corrente = get_stagione_corrente()
    if stagione_corrente:
        system_prompt += (
            f" La stagione più recente disponibile nel database è {stagione_corrente}: "
            f"quando l'utente dice 'ultima stagione', 'quest'anno' o 'stagione corrente' "
            f"intende sempre questa."
        )

    messages = (
        [{"role": "system", "content": system_prompt}]
        + req.storico
        + [{"role": "user", "content": req.domanda}]
    )

    contesto_usato: list[str] = []

    # Primo giro: il modello sceglie lo strumento (puo' anche chiamarne piu' di uno)
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
            elif tool_call.function.name == "stato_mio_budget":
                risultato = esegui_mio_budget()
                contesto_usato.append(
                    f"Mio budget: {risultato.get('budget_residuo')} residui su {risultato.get('budget_totale')}"
                )
            elif tool_call.function.name == "la_mia_rosa":
                righe = esegui_mia_rosa()
                risultato = [dict(r) for r in righe]
                if risultato:
                    contesto_usato.extend(
                        f"{r['nome']} ({r['ruolo']}, {r['squadra']}), pagato {r['prezzo_pagato']}"
                        for r in risultato
                    )
                else:
                    contesto_usato.append("Nessun giocatore ancora in rosa.")
            elif tool_call.function.name == "rosa_di_squadra":
                righe = esegui_rosa_di_squadra(**args)
                risultato = [dict(r) for r in righe]
                if risultato:
                    nome_reale = risultato[0]["nome_squadra"]
                    allenatore = risultato[0]["allenatore"]
                    contesto_usato.append(f"Squadra trovata: '{nome_reale}' (allenatore: {allenatore})")
                    contesto_usato.extend(
                        f"{r['nome']} ({r['ruolo']}, {r['squadra']}), pagato {r['prezzo_finale']}"
                        for r in risultato
                    )
                else:
                    contesto_usato.append(f"Nessuna squadra trovata per '{args.get('nome_squadra')}'.")
            elif tool_call.function.name == "strategia_asta":
                risultato = esegui_strategia_asta()
                contesto_usato.append(json.dumps(risultato, default=str, ensure_ascii=False))
            elif tool_call.function.name == "cerca_statistiche_storiche":
                righe = esegui_statistiche_storiche(**args)
                risultato = [dict(r) for r in righe]
                contesto_usato.extend(
                    f"{r['nome']} ({r['ruolo']}, {r['squadra']}) "
                    f"stagione {r['stagione']}: "
                    f"fantamedia {r['fantamedia']}, gol {r['gol']}, assist {r['assist']}, "
                    f"presenze {r['presenze']}"
                    for r in risultato
                )
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

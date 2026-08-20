"""
FantAssistant Databricks — Auction Tracker App (FastAPI)

Sostituisce interamente services/auction-tracker/main.py.
Differenze rispetto alla versione Podman:
  - psycopg (PostgreSQL)  -> databricks-sql-connector
  - %s placeholder         -> ?
  - Nomi tabella           -> `catalog`.`schema`.tabella
  - ON CONFLICT (upsert)   -> MERGE INTO Delta Lake
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.sql import StatementState, StatementParameterListItem

_cfg = Config()  # usa automaticamente l'auth del service principal dell'app
_ws = WorkspaceClient(config=_cfg)

DATABRICKS_HTTP_PATH = os.environ["DATABRICKS_SQL_HTTP_PATH"]
CATALOG = os.environ.get("UNITY_CATALOG", "platform")  # tabelle in catalog 'platform'
SCHEMA  = os.environ.get("UNITY_SCHEMA",  "fantassistant")
NS      = f"`{CATALOG}`.`{SCHEMA}`"
_WAREHOUSE_ID = DATABRICKS_HTTP_PATH.rstrip("/").split("/")[-1]

app = FastAPI(title="FantAssistant Auction Tracker — Databricks")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
    cols = [c.name for c in response.manifest.schema.columns]
    rows = response.result.data_array or []
    return [dict(zip(cols, row)) for row in rows]


def execute_dml(sql: str, params: dict[str, str] | None = None):
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


class RegistraAcquisto(BaseModel):
    giocatore_id: int
    prezzo_finale: float
    squadra_acquirente: str
    e_mio: bool = False
    fonte: str = "manuale"


class SquadraInit(BaseModel):
    nome: str
    allenatore: str | None = None


class InitSquadre(BaseModel):
    squadre: list[SquadraInit]
    budget_totale: float


@app.get("/health")
def health():
    return {"status": "ok", "backend": "databricks"}


@app.post("/squadre/init")
def init_squadre(req: InitSquadre):
    """
    Censimento iniziale: registra tutte le squadre della lega.
    Usa MERGE INTO per essere idempotente (equivalente di ON CONFLICT DO UPDATE).
    """
    for s in req.squadre:
        nome      = s.nome.strip()
        allenatore = s.allenatore.strip() if s.allenatore else None
        execute_dml(f"""
            MERGE INTO {NS}.squadre AS target
            USING (SELECT :nome AS nome) AS src
            ON target.nome = src.nome
            WHEN MATCHED THEN
                UPDATE SET allenatore = :allenatore, budget_totale = :budget_totale
            WHEN NOT MATCHED THEN
                INSERT (nome, allenatore, budget_totale)
                VALUES (:nome, :allenatore, :budget_totale)
        """, {"nome": nome, "allenatore": allenatore or "", "budget_totale": req.budget_totale})
    return {"status": "ok", "squadre_registrate": len(req.squadre)}


@app.get("/squadre")
def lista_squadre():
    righe = query_rows(f"""
        SELECT s.nome, s.allenatore, s.budget_totale,
               COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
        FROM {NS}.squadre s
        LEFT JOIN {NS}.asta_log a ON a.squadra_acquirente = s.nome
        GROUP BY s.nome, s.allenatore, s.budget_totale
        ORDER BY s.nome
    """)
    return {
        "squadre": [
            {**r, "budget_residuo": float(r["budget_totale"]) - float(r["budget_speso"])}
            for r in righe
        ]
    }


@app.post("/acquisto")
def registra_acquisto(req: RegistraAcquisto):
    # Verifica squadra censita e budget residuo
    squadra_rows = query_rows(f"""
        SELECT s.budget_totale,
               COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
        FROM {NS}.squadre s
        LEFT JOIN {NS}.asta_log a ON a.squadra_acquirente = s.nome
        WHERE s.nome = :squadra_nome
        GROUP BY s.budget_totale
    """, {"squadra_nome": req.squadra_acquirente})

    if not squadra_rows:
        raise HTTPException(
            400,
            f"Squadra '{req.squadra_acquirente}' non censita. "
            "Registrala prima con POST /squadre/init.",
        )

    squadra = squadra_rows[0]
    residuo = float(squadra["budget_totale"]) - float(squadra["budget_speso"])
    if req.prezzo_finale > residuo:
        raise HTTPException(
            400,
            f"Budget insufficiente per '{req.squadra_acquirente}': "
            f"residuo {residuo}, richiesti {req.prezzo_finale}.",
        )

    execute_dml(f"""
        INSERT INTO {NS}.asta_log (giocatore_id, prezzo_finale, squadra_acquirente, fonte)
        VALUES (:giocatore_id, :prezzo_finale, :squadra_acquirente, :fonte)
    """, {"giocatore_id": req.giocatore_id, "prezzo_finale": req.prezzo_finale, "squadra_acquirente": req.squadra_acquirente, "fonte": req.fonte})

    if req.e_mio:
        ruolo_rows = query_rows(f"""
            SELECT ruolo FROM {NS}.giocatori WHERE id = :giocatore_id
        """, {"giocatore_id": req.giocatore_id})
        if not ruolo_rows:
            raise HTTPException(404, "Giocatore non trovato")
        ruolo = ruolo_rows[0]["ruolo"]
        execute_dml(f"""
            INSERT INTO {NS}.mia_rosa (giocatore_id, prezzo_pagato, ruolo_fanta)
            VALUES (:giocatore_id, :prezzo_pagato, :ruolo_fanta)
        """, {"giocatore_id": req.giocatore_id, "prezzo_pagato": req.prezzo_finale, "ruolo_fanta": ruolo})

    return {"status": "ok"}


@app.get("/giocatori-liberi")
def giocatori_liberi(ruolo: str | None = None):
    sql = f"""
        SELECT g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale
        FROM {NS}.giocatori g
        WHERE g.id NOT IN (
            SELECT giocatore_id FROM {NS}.asta_log WHERE giocatore_id IS NOT NULL
        )
    """
    params = {}
    if ruolo:
        sql += " AND g.ruolo = :ruolo"
        params["ruolo"] = ruolo
    return {"giocatori_liberi": query_rows(sql, params or None)}


@app.get("/cerca-giocatori")
def cerca_giocatori(nome: str, solo_liberi: bool = True):
    sql = f"""
        SELECT g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale, g.fvm
        FROM {NS}.giocatori g
        WHERE lower(g.nome) LIKE lower(:nome_pattern)
    """
    params = {"nome_pattern": f"%{nome}%"}
    if solo_liberi:
        sql += f" AND g.id NOT IN (SELECT giocatore_id FROM {NS}.asta_log WHERE giocatore_id IS NOT NULL)"
    sql += " ORDER BY g.fvm DESC NULLS LAST LIMIT 10"
    return {"risultati": query_rows(sql, params or None)}


@app.get("/storico")
def storico(limit: int = 20):
    rows = query_rows(f"""
        SELECT a.id, g.nome, g.ruolo, g.squadra, a.prezzo_finale,
               a.squadra_acquirente, a.fonte, a.creato_il
        FROM {NS}.asta_log a
        JOIN {NS}.giocatori g ON g.id = a.giocatore_id
        ORDER BY a.creato_il DESC
        LIMIT {limit}
    """)
    return {"storico": rows}

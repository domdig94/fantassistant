"""
FantAssistant - Auction tracker
Nessuna AI qui: solo logica di budget e disponibilita' giocatori durante l'asta.
Si integra con il servizio vision (che scrive su asta_log) e serve a rispondere
in tempo reale a "quanto mi resta", "chi e' ancora libero", ecc.
"""
import os

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DATABASE_URL = os.environ["DATABASE_URL"]

app = FastAPI(title="FantAssistant Auction Tracker")


def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


class ImpostaBudget(BaseModel):
    ruolo: str
    budget_totale: float


class RegistraAcquisto(BaseModel):
    giocatore_id: int
    prezzo_finale: float
    squadra_acquirente: str
    e_mio: bool = False
    fonte: str = "manuale"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/budget")
def imposta_budget(req: ImpostaBudget):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO budget_tracker (ruolo, budget_totale, budget_speso)
                VALUES (%s, %s, 0)
                """,
                (req.ruolo, req.budget_totale),
            )
            conn.commit()
    return {"status": "ok"}


@app.get("/budget")
def leggi_budget():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ruolo, budget_totale, budget_speso FROM budget_tracker")
            righe = cur.fetchall()
    return {
        "budget": [
            {**r, "budget_residuo": float(r["budget_totale"]) - float(r["budget_speso"])}
            for r in righe
        ]
    }


@app.post("/acquisto")
def registra_acquisto(req: RegistraAcquisto):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO asta_log (giocatore_id, prezzo_finale, squadra_acquirente, fonte)
                VALUES (%s, %s, %s, %s)
                """,
                (req.giocatore_id, req.prezzo_finale, req.squadra_acquirente, req.fonte),
            )

            if req.e_mio:
                cur.execute("SELECT ruolo FROM giocatori WHERE id = %s", (req.giocatore_id,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, "Giocatore non trovato")

                cur.execute(
                    """
                    INSERT INTO mia_rosa (giocatore_id, prezzo_pagato, ruolo_fanta)
                    VALUES (%s, %s, %s)
                    """,
                    (req.giocatore_id, req.prezzo_finale, row["ruolo"]),
                )
                cur.execute(
                    """
                    UPDATE budget_tracker
                    SET budget_speso = budget_speso + %s
                    WHERE ruolo = %s
                    """,
                    (req.prezzo_finale, row["ruolo"]),
                )
            conn.commit()
    return {"status": "ok"}


@app.get("/giocatori-liberi")
def giocatori_liberi(ruolo: str | None = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale
                FROM giocatori g
                WHERE g.id NOT IN (SELECT giocatore_id FROM asta_log)
            """
            params = ()
            if ruolo:
                query += " AND g.ruolo = %s"
                params = (ruolo,)
            cur.execute(query, params)
            righe = cur.fetchall()
    return {"giocatori_liberi": righe}

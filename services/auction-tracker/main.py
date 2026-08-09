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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATABASE_URL = os.environ["DATABASE_URL"]

app = FastAPI(title="FantAssistant Auction Tracker")

# Il frontend gira come pagina statica su un'origine diversa (nginx su
# un'altra porta): serve CORS aperto per un tool interno d'uso personale.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


class RegistraAcquisto(BaseModel):
    giocatore_id: int
    prezzo_finale: float
    squadra_acquirente: str
    e_mio: bool = False
    fonte: str = "manuale"

class InitSquadre(BaseModel):
    nomi: list[str]
    budget_totale: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/squadre/init")
def init_squadre(req: InitSquadre):
    """Censimento iniziale: registra tutte le squadre della lega con lo
    stesso budget di partenza. Rilanciabile senza duplicare (upsert)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for nome in req.nomi:
                cur.execute(
                    """
                    INSERT INTO squadre (nome, budget_totale)
                    VALUES (%s, %s)
                    ON CONFLICT (nome) DO UPDATE SET budget_totale = EXCLUDED.budget_totale
                    """,
                    (nome.strip(), req.budget_totale),
                )
            conn.commit()
    return {"status": "ok", "squadre_registrate": len(req.nomi)}


@app.get("/squadre")
def lista_squadre():
    """Budget totale, speso (somma da asta_log) e residuo per ogni squadra."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT s.nome, s.budget_totale,
                       COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
                FROM squadre s
                LEFT JOIN asta_log a ON a.squadra_acquirente = s.nome
                GROUP BY s.nome, s.budget_totale
                ORDER BY s.nome
            """)
            righe = cur.fetchall()
    return {
        "squadre": [
            {**r, "budget_residuo": float(r["budget_totale"]) - float(r["budget_speso"])}
            for r in righe
        ]
    }


@app.post("/acquisto")
def registra_acquisto(req: RegistraAcquisto):
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Verifica che la squadra sia censita e ricava il budget residuo
            # prima di scrivere qualsiasi cosa - controllo bloccante, non solo
            # un warning: durante un'asta vera un errore qui e' irreversibile.
            cur.execute(
                """
                SELECT s.budget_totale,
                       COALESCE(SUM(a.prezzo_finale), 0) AS budget_speso
                FROM squadre s
                LEFT JOIN asta_log a ON a.squadra_acquirente = s.nome
                WHERE s.nome = %s
                GROUP BY s.budget_totale
                """,
                (req.squadra_acquirente,),
            )
            squadra = cur.fetchone()

            if not squadra:
                raise HTTPException(
                    400,
                    f"Squadra '{req.squadra_acquirente}' non censita. "
                    f"Registrala prima con POST /squadre/init.",
                )

            residuo = float(squadra["budget_totale"]) - float(squadra["budget_speso"])
            if req.prezzo_finale > residuo:
                raise HTTPException(
                    400,
                    f"Budget insufficiente per '{req.squadra_acquirente}': "
                    f"residuo {residuo}, richiesti {req.prezzo_finale}.",
                )

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


@app.get("/cerca-giocatori")
def cerca_giocatori(nome: str, solo_liberi: bool = True):
    """Autocomplete per il form live: cerca per nome (match parziale),
    di default solo tra i giocatori non ancora assegnati in asta."""
    query = """
        SELECT g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale, g.fvm
        FROM giocatori g
        WHERE g.nome ILIKE %s
    """
    params = [f"%{nome}%"]
    if solo_liberi:
        query += " AND g.id NOT IN (SELECT giocatore_id FROM asta_log)"
    query += " ORDER BY g.fvm DESC NULLS LAST LIMIT 10"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return {"risultati": cur.fetchall()}


@app.get("/storico")
def storico(limit: int = 20):
    """Ultimi acquisti registrati, con nome giocatore, per il log live."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id, g.nome, g.ruolo, g.squadra, a.prezzo_finale,
                       a.squadra_acquirente, a.fonte, a.creato_il
                FROM asta_log a
                JOIN giocatori g ON g.id = a.giocatore_id
                ORDER BY a.creato_il DESC
                LIMIT %s
                """,
                (limit,),
            )
            return {"storico": cur.fetchall()}
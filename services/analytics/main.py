"""
FantAssistant - Analytics Service

Endpoint REST per analisi e statistiche a supporto di:
  - Asta live: valutazione giocatori, confronto, valore atteso, tendenza mercato
  - Day-by-day: rendimento rosa, classifica lega, top/flop giornata

Porta: 8004
"""
import os
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

DATABASE_URL = os.environ["DATABASE_URL"]
MY_TEAM = os.environ.get("MY_TEAM", "Io")
# FVM nel listone ufficiale è calibrato su budget 1000.
# BUDGET_LEGA permette di normalizzare: fvm_norm = fvm * (BUDGET_LEGA / 1000)
BUDGET_LEGA = int(os.environ.get("BUDGET_LEGA", "1000"))

app = FastAPI(title="FantAssistant Analytics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# RICERCA GIOCATORI  (entry point testuale per l'utente)
# ---------------------------------------------------------------------------

@app.get("/cerca")
def cerca(q: str = Query(..., min_length=2, description="Nome o parte del nome del giocatore")):
    """
    Ricerca giocatori per nome (match parziale, case-insensitive).
    Ritorna id, nome, ruolo, squadra, quotazione_attuale, fvm.
    Usato dal frontend come autocomplete prima di chiamare /giocatore/{id}/scheda.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, nome, ruolo, squadra, quotazione_attuale, fvm
                FROM giocatori
                WHERE nome ILIKE %s
                ORDER BY nome
                LIMIT 20
                """,
                (f"%{q}%",),
            )
            return {"risultati": [dict(r) for r in cur.fetchall()]}


# ---------------------------------------------------------------------------
# SCHEDA GIOCATORE
# ---------------------------------------------------------------------------

@app.get("/giocatore/{giocatore_id}/scheda")
def scheda_giocatore(giocatore_id: int):
    """
    Profilo completo di un giocatore:
    - Anagrafica (quotazione, FVM, ruolo, squadra)
    - Storico statistiche per stagione
    - Trend fantamedia (per grafico a linee)
    - Riepilogo bonus/malus aggregato (per radar chart)
    - Flag se e' nella rosa dell'utente e prezzo pagato
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.id, g.nome, g.ruolo, g.squadra,
                       g.quotazione_attuale, g.quotazione_iniziale, g.fvm,
                       mr.prezzo_pagato,
                       EXISTS(SELECT 1 FROM mia_rosa mr2 WHERE mr2.giocatore_id = g.id) AS in_rosa
                FROM giocatori g
                LEFT JOIN mia_rosa mr ON mr.giocatore_id = g.id
                WHERE g.id = %s
                """,
                (giocatore_id,),
            )
            g = cur.fetchone()
            if not g:
                raise HTTPException(status_code=404, detail="Giocatore non trovato")

            cur.execute(
                """
                SELECT stagione, presenze, media_voto, fantamedia,
                       gol, assist, ammonizioni, espulsioni,
                       gol_subiti, rigori_parati, rigori_calciati,
                       bonus, malus, autogol
                FROM statistiche_storiche
                WHERE giocatore_id = %s
                ORDER BY stagione ASC
                """,
                (giocatore_id,),
            )
            stagioni = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT giornata, voto, fantavoto, gol, assist,
                       ammonizione, espulsione, stagione
                FROM voti_giornata
                WHERE giocatore_id = %s
                ORDER BY stagione DESC, giornata ASC
                LIMIT 38
                """,
                (giocatore_id,),
            )
            voti = [dict(r) for r in cur.fetchall()]

    trend_fantamedia = {
        "labels": [s["stagione"] for s in stagioni],
        "values": [float(s["fantamedia"]) if s["fantamedia"] is not None else None for s in stagioni],
    }

    def somma(campo):
        return sum((s[campo] or 0) for s in stagioni)

    radar = {
        "labels": ["Gol", "Assist", "Rigori parati", "Rigori calciati", "Ammonizioni", "Espulsioni", "Autogol"],
        "values": [
            somma("gol"),
            somma("assist"),
            somma("rigori_parati"),
            somma("rigori_calciati"),
            somma("ammonizioni"),
            somma("espulsioni"),
            somma("autogol"),
        ],
    }

    return {
        "giocatore": dict(g),
        "stagioni": stagioni,
        "trend_fantamedia": trend_fantamedia,
        "radar_bonus_malus": radar,
        "voti_giornata": voti,
    }


# ---------------------------------------------------------------------------
# CONFRONTO GIOCATORI
# ---------------------------------------------------------------------------

@app.get("/confronta")
def confronta(ids: str = Query(..., description="Id giocatori separati da virgola, es. 1,2,3")):
    """
    Confronto side-by-side tra piu' giocatori (max 4).
    Per ogni giocatore ritorna anagrafica + medie aggregate sulle stagioni disponibili.
    """
    try:
        id_list = [int(i.strip()) for i in ids.split(",")][:4]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids deve essere una lista di interi separati da virgola")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.id, g.nome, g.ruolo, g.squadra,
                       g.quotazione_attuale, g.fvm,
                       ROUND(AVG(s.fantamedia)::numeric, 2)   AS fantamedia_media,
                       ROUND(AVG(s.media_voto)::numeric, 2)   AS media_voto_media,
                       ROUND(AVG(s.presenze)::numeric, 1)     AS presenze_media,
                       ROUND(AVG(s.gol)::numeric, 1)          AS gol_media,
                       ROUND(AVG(s.assist)::numeric, 1)       AS assist_media,
                       ROUND(AVG(s.ammonizioni)::numeric, 1)  AS ammonizioni_media,
                       COUNT(s.stagione)                      AS stagioni_disponibili
                FROM giocatori g
                LEFT JOIN statistiche_storiche s ON s.giocatore_id = g.id
                WHERE g.id = ANY(%s)
                GROUP BY g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale, g.fvm
                ORDER BY fantamedia_media DESC NULLS LAST
                """,
                (id_list,),
            )
            giocatori = [dict(r) for r in cur.fetchall()]

    def dataset(campo):
        return [
            {
                "label": f"{g['nome']} ({g['squadra']})",
                "data": [float(g[campo]) if g[campo] is not None else 0],
            }
            for g in giocatori
        ]

    nomi = [f"{g['nome']} ({g['squadra']})" for g in giocatori]

    # Grafico 1: Fantamedia e Media voto (scala 0-10)
    chart_voti = {
        "labels": nomi,
        "datasets": [
            {"label": "Fantamedia",  "data": [float(g["fantamedia_media"]  or 0) for g in giocatori]},
            {"label": "Media voto",  "data": [float(g["media_voto_media"]  or 0) for g in giocatori]},
        ],
    }

    # Grafico 3: Presenze (scala 0-38)
    chart_presenze = {
        "labels": nomi,
        "datasets": [
            {"label": "Presenze", "data": [float(g["presenze_media"] or 0) for g in giocatori]},
        ],
    }
    # Grafico 2: Gol, Assist, Ammonizioni — una metrica per dataset,
    # labels = nomi giocatori, così le barre di ogni giocatore sono affiancate
    # per ciascuna metrica sull'asse X.
    # Struttura: asse X = metriche, un dataset per giocatore.
    chart_bonus = {
        "labels": ["Gol", "Assist", "Ammonizioni"],
        "datasets": [
            {
                "label": f"{g['nome']} ({g['squadra']})",
                "data": [
                    float(g["gol_media"]        or 0),
                    float(g["assist_media"]      or 0),
                    float(g["ammonizioni_media"] or 0),
                ],
            }
            for g in giocatori
        ],
    }

    return {
        "giocatori": giocatori,
        "chart_voti": chart_voti,
        "chart_bonus": chart_bonus,
        "chart_presenze": chart_presenze,
    }


# ---------------------------------------------------------------------------
# VALORE ATTESO  (supporto asta)
# ---------------------------------------------------------------------------

@app.get("/valore-atteso")
def valore_atteso(
    ruolo: Optional[str] = Query(None, description="P, D, C o A"),
    budget_max: Optional[int] = Query(None, description="Budget massimo da spendere"),
    limit: int = Query(20, le=50),
):
    """
    Score 0-100 per i giocatori liberi (non ancora assegnati in asta).

    Formula:
      score = (fvm_norm * 0.60 + fm_pond_norm * 0.40) * affidabilita * 100

    Componenti:
      - fvm_norm: FVM del giocatore normalizzato 0-1 rispetto al massimo del suo ruolo
        (usa il FVM della stagione corrente, gia' scalato al budget lega in fase di import)
      - fm_pond_norm: fantamedia ponderata normalizzata 0-1 rispetto al massimo del ruolo.
        La fantamedia ponderata e' calcolata sulle ultime 4 stagioni disponibili con pesi
        decrescenti [3, 2, 1, 0.5] (stagione piu' recente = peso maggiore), cosi' il
        rendimento recente conta di piu' senza ignorare la storia.
      - affidabilita: fattore 0-1 basato sulle presenze medie ponderate (stessi pesi).
        Serve a penalizzare chi ha poche presenze (dati storici non affidabili):
          < 10 presenze -> 0.30
          10-19         -> 0.60
          20-27         -> 0.85
          28+           -> 1.00

    Pesi FVM/fantamedia: 60% FVM (forward-looking, stima piattaforma stagione corrente)
                         40% fantamedia ponderata (backward-looking, prova del campo reale)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Recupera le ultime 4 stagioni disponibili per ogni giocatore libero,
            # con tutti i dati necessari per calcolare lo score in Python.
            # Il calcolo della media ponderata con pesi posizionali non e' esprimibile
            # in modo pulito in SQL standard, quindi si fa in Python.
            params: list = []
            query = """
                SELECT g.id, g.nome, g.ruolo, g.squadra,
                       g.quotazione_attuale, g.fvm,
                       s.stagione, s.fantamedia, s.presenze
                FROM giocatori g
                LEFT JOIN (
                    SELECT giocatore_id, stagione, fantamedia, presenze,
                           ROW_NUMBER() OVER (
                               PARTITION BY giocatore_id ORDER BY stagione DESC
                           ) AS rn
                    FROM statistiche_storiche
                ) s ON s.giocatore_id = g.id AND s.rn <= 4
                WHERE g.id NOT IN (SELECT giocatore_id FROM asta_log)
            """
            if ruolo:
                query += " AND g.ruolo = %s"
                params.append(ruolo)
            if budget_max:
                query += " AND g.fvm <= %s"
                params.append(budget_max)
            query += " ORDER BY g.id, s.stagione DESC"
            cur.execute(query, params)
            rows = cur.fetchall()

    # --- Aggrega per giocatore e calcola fantamedia ponderata ---
    PESI = [3, 2, 1, 0.5]  # indice 0 = stagione più recente

    def affidabilita(presenze_medie_pond: float) -> float:
        if presenze_medie_pond < 10:  return 0.30
        if presenze_medie_pond < 20:  return 0.60
        if presenze_medie_pond < 28:  return 0.85
        return 1.00

    giocatori_map: dict = {}
    for r in rows:
        gid = r["id"]
        if gid not in giocatori_map:
            giocatori_map[gid] = {
                "id": r["id"], "nome": r["nome"], "ruolo": r["ruolo"],
                "squadra": r["squadra"], "quotazione_attuale": float(r["quotazione_attuale"] or 0),
                "fvm": float(r["fvm"] or 0), "stagioni": [],
            }
        if r["stagione"] is not None:
            giocatori_map[gid]["stagioni"].append({
                "stagione": r["stagione"],
                "fantamedia": float(r["fantamedia"] or 0),
                "presenze": float(r["presenze"] or 0),
            })

    risultati = []
    for g in giocatori_map.values():
        stagioni = g["stagioni"]  # già ordinate DESC per stagione
        n = len(stagioni)
        pesi_usati = PESI[:n]
        somma_pesi = sum(pesi_usati)

        if n > 0 and somma_pesi > 0:
            fm_pond = sum(
                float(stagioni[i]["fantamedia"] or 0) * pesi_usati[i]
                for i in range(n)
            ) / somma_pesi
            pres_pond = sum(
                float(stagioni[i]["presenze"] or 0) * pesi_usati[i]
                for i in range(n)
            ) / somma_pesi
            n_stagioni = n
        else:
            fm_pond = None
            pres_pond = 0
            n_stagioni = 0

        risultati.append({
            **{k: v for k, v in g.items() if k != "stagioni"},
            "fantamedia_ponderata": round(fm_pond, 2) if fm_pond is not None else None,
            "presenze_medie_pond": round(pres_pond, 1),
            "affidabilita": affidabilita(pres_pond),
            "n_stagioni": n_stagioni,
            "stagioni_dettaglio": stagioni,
            # score calcolato dopo normalizzazione
            "score": None,
        })

    # --- Normalizza FVM e fantamedia ponderata per ruolo (0-1) ---
    ruoli_presenti = set(g["ruolo"] for g in risultati)
    for ruolo_corrente in ruoli_presenti:
        subset = [g for g in risultati if g["ruolo"] == ruolo_corrente]

        max_fvm = max((g["fvm"] or 0) for g in subset) or 1
        max_fm  = max((g["fantamedia_ponderata"] or 0) for g in subset) or 1

        for g in subset:
            fvm_norm = (g["fvm"] or 0) / max_fvm
            fm_norm  = (g["fantamedia_ponderata"] or 0) / max_fm
            aff      = g["affidabilita"]
            g["score"] = round((fvm_norm * 0.60 + fm_norm * 0.40) * aff * 100, 1)

    # Ordina per score desc, applica limit
    risultati.sort(key=lambda g: g["score"] or 0, reverse=True)
    risultati = risultati[:limit]

    return {"giocatori": risultati}


# ---------------------------------------------------------------------------
# TENDENZA MERCATO  (supporto asta)
# ---------------------------------------------------------------------------

@app.get("/tendenza-mercato")
def tendenza_mercato():
    """
    Per ogni ruolo: prezzo medio pagato in asta vs FVM medio.
    Mostra se il mercato sta sopravvalutando o sottovalutando certi ruoli.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.ruolo,
                       COUNT(*)                                          AS acquisti,
                       ROUND(AVG(a.prezzo_finale)::numeric, 1)          AS prezzo_medio_pagato,
                       ROUND(AVG(g.fvm)::numeric, 1)                    AS fvm_medio,
                       ROUND(AVG(stat.fantamedia_media)::numeric, 2)    AS fantamedia_storica_media,
                       ROUND(
                           (AVG(a.prezzo_finale) / NULLIF(AVG(g.fvm), 0))::numeric, 2
                       )                                                 AS rapporto_prezzo_fvm
                FROM asta_log a
                JOIN giocatori g ON g.id = a.giocatore_id
                LEFT JOIN (
                    SELECT giocatore_id, AVG(fantamedia) AS fantamedia_media
                    FROM statistiche_storiche
                    GROUP BY giocatore_id
                ) stat ON stat.giocatore_id = g.id
                GROUP BY g.ruolo
                ORDER BY g.ruolo
                """
            )
            ruoli = [dict(r) for r in cur.fetchall()]

    # FVM già scalato al budget lega in fase di import — uso diretto
    etichette_ruolo = {"P": "Portieri", "D": "Difensori", "C": "Centrocampisti", "A": "Attaccanti"}
    risultati = []
    for r in ruoli:
        prezzo = float(r["prezzo_medio_pagato"] or 0)
        fvm = float(r["fvm_medio"] or 0)
        rapporto = round(prezzo / fvm, 2) if fvm > 0 else None

        if rapporto is None:
            valutazione = "Dati insufficienti"
            colore = "gray"
        elif rapporto > 1.15:
            valutazione = f"Il mercato sta pagando {round((rapporto-1)*100)}% sopra il valore atteso — ruolo gonfiato"
            colore = "red"
        elif rapporto < 0.85:
            valutazione = f"Il mercato sta pagando {round((1-rapporto)*100)}% sotto il valore atteso — buone occasioni"
            colore = "green"
        else:
            valutazione = "Prezzi in linea con il valore atteso"
            colore = "amber"

        risultati.append({
            **dict(r),
            "fvm_medio_norm": fvm,
            "rapporto_prezzo_fvm_norm": rapporto,
            "valutazione": valutazione,
            "colore": colore,
            "etichetta_ruolo": etichette_ruolo.get(r["ruolo"], r["ruolo"]),
        })

    # Chart 1: prezzo medio vs FVM (già sulla scala del budget lega)
    chart_prezzi = {
        "labels": [r["etichetta_ruolo"] for r in risultati],
        "datasets": [
            {
                "label": "Prezzo medio pagato",
                "data": [float(r["prezzo_medio_pagato"] or 0) for r in risultati],
            },
            {
                "label": f"FVM medio (scala {BUDGET_LEGA})",
                "data": [r["fvm_medio_norm"] for r in risultati],
            },
        ],
    }

    # Chart 2: rapporto prezzo/FVM per ruolo (1.0 = perfettamente in linea)
    chart_rapporto = {
        "labels": [r["etichetta_ruolo"] for r in risultati],
        "datasets": [
            {
                "label": "Rapporto prezzo / FVM (1.0 = in linea)",
                "data": [r["rapporto_prezzo_fvm_norm"] for r in risultati],
            }
        ],
        "soglia": 1.0,
    }

    return {
        "budget_lega": BUDGET_LEGA,
        "per_ruolo": risultati,
        "chart_prezzi": chart_prezzi,
        "chart_rapporto": chart_rapporto,
    }


# ---------------------------------------------------------------------------
# RENDIMENTO MIA ROSA  (day-by-day)
# ---------------------------------------------------------------------------

@app.get("/mia-rosa/rendimento")
def rendimento_rosa():
    """
    Per ogni giocatore nella mia rosa:
    - Fantamedia storica aggregata
    - Voti nelle giornate della stagione corrente
    - Trend per grafico a linee
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.id, g.nome, g.ruolo, g.squadra, mr.prezzo_pagato,
                       ROUND(AVG(s.fantamedia)::numeric, 2) AS fantamedia_storica
                FROM mia_rosa mr
                JOIN giocatori g ON g.id = mr.giocatore_id
                LEFT JOIN statistiche_storiche s ON s.giocatore_id = g.id
                GROUP BY g.id, g.nome, g.ruolo, g.squadra, mr.prezzo_pagato
                ORDER BY g.ruolo, fantamedia_storica DESC NULLS LAST
                """
            )
            rosa = [dict(r) for r in cur.fetchall()]

            if rosa:
                ids_rosa = [r["id"] for r in rosa]
                cur.execute(
                    """
                    SELECT giocatore_id, giornata, fantavoto
                    FROM voti_giornata
                    WHERE giocatore_id = ANY(%s)
                    ORDER BY giornata ASC
                    """,
                    (ids_rosa,),
                )
                voti_rows = cur.fetchall()
            else:
                voti_rows = []

    voti_per_giocatore: dict = {}
    for v in voti_rows:
        gid = v["giocatore_id"]
        voti_per_giocatore.setdefault(gid, []).append(
            {"giornata": v["giornata"], "fantavoto": float(v["fantavoto"]) if v["fantavoto"] else None}
        )

    giornate = sorted({v["giornata"] for v in voti_rows}) if voti_rows else []
    chart_trend = {
        "labels": [f"G{g}" for g in giornate],
        "datasets": [
            {
                "label": r["nome"],
                "data": [
                    next(
                        (v["fantavoto"] for v in voti_per_giocatore.get(r["id"], []) if v["giornata"] == g),
                        None,
                    )
                    for g in giornate
                ],
            }
            for r in rosa
        ],
    }

    return {
        "rosa": [
            {**r, "voti_giornata": voti_per_giocatore.get(r["id"], [])}
            for r in rosa
        ],
        "chart_trend": chart_trend,
    }


# ---------------------------------------------------------------------------
# CLASSIFICA LEGA  (day-by-day)
# ---------------------------------------------------------------------------

@app.get("/classifica-lega")
def classifica_lega():
    """
    Punteggio totale per squadra della lega dai voti_giornata.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.squadra_acquirente AS squadra,
                       COUNT(DISTINCT vg.giornata)              AS giornate_giocate,
                       ROUND(SUM(vg.fantavoto)::numeric, 1)     AS punti_totali,
                       ROUND(AVG(vg.fantavoto)::numeric, 2)     AS fantamedia_squadra
                FROM asta_log a
                JOIN voti_giornata vg ON vg.giocatore_id = a.giocatore_id
                GROUP BY a.squadra_acquirente
                ORDER BY punti_totali DESC NULLS LAST
                """
            )
            classifica = [dict(r) for r in cur.fetchall()]

    chart = {
        "labels": [r["squadra"] for r in classifica],
        "datasets": [
            {
                "label": "Punti totali",
                "data": [float(r["punti_totali"] or 0) for r in classifica],
            }
        ],
    }

    return {"classifica": classifica, "chart": chart}


# ---------------------------------------------------------------------------
# TOP/FLOP GIORNATA  (day-by-day)
# ---------------------------------------------------------------------------

@app.get("/giornata/{numero}/top")
def top_giornata(numero: int, limit: int = Query(10, le=20)):
    """
    Migliori e peggiori giocatori di una giornata specifica.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.nome, g.ruolo, g.squadra,
                       vg.voto, vg.fantavoto,
                       vg.gol, vg.assist, vg.ammonizione, vg.espulsione,
                       a.squadra_acquirente AS squadra_fanta
                FROM voti_giornata vg
                JOIN giocatori g ON g.id = vg.giocatore_id
                LEFT JOIN asta_log a ON a.giocatore_id = vg.giocatore_id
                WHERE vg.giornata = %s
                ORDER BY vg.fantavoto DESC NULLS LAST
                """,
                (numero,),
            )
            tutti = [dict(r) for r in cur.fetchall()]

    top = tutti[:limit]
    flop = sorted(tutti, key=lambda x: (x["fantavoto"] or 0))[:limit]

    return {"giornata": numero, "top": top, "flop": flop}

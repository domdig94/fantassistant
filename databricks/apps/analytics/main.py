"""
FantAssistant Databricks — Analytics App (FastAPI)

Sostituisce interamente services/analytics/main.py.
Differenze rispetto alla versione Podman:
  - psycopg (PostgreSQL)  -> databricks-sql-connector
  - %s placeholder         -> ?
  - Nomi tabella           -> `catalog`.`schema`.tabella
  - ANY(%s) PostgreSQL     -> IN (...) con join o subquery compatibile Spark SQL
  - ILIKE                  -> lower() LIKE lower()
  - ::numeric cast         -> ROUND() nativo (Spark SQL lo supporta)
"""
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from databricks import sql as dbsql

from databricks.sdk.core import Config

_cfg = Config()  # usa automaticamente l'auth del service principal dell'app

DATABRICKS_HOST      = _cfg.host
DATABRICKS_HTTP_PATH = os.environ["DATABRICKS_SQL_HTTP_PATH"]
CATALOG    = os.environ.get("UNITY_CATALOG", "fantassistant")
SCHEMA     = os.environ.get("UNITY_SCHEMA",  "main")
NS         = f"`{CATALOG}`.`{SCHEMA}`"
MY_TEAM    = os.environ.get("MY_TEAM", "Io")
BUDGET_LEGA = int(os.environ.get("BUDGET_LEGA", "1000"))

app = FastAPI(title="FantAssistant Analytics — Databricks")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_conn():
    return dbsql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", ""),
        http_path=DATABRICKS_HTTP_PATH,
        credentials_provider=_cfg.authenticate,
    )


def query_rows(sql: str, params: list | None = None) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "backend": "databricks"}


# ---------------------------------------------------------------------------
# RICERCA GIOCATORI
# ---------------------------------------------------------------------------

@app.get("/cerca")
def cerca(q: str = Query(..., min_length=2)):
    rows = query_rows(f"""
        SELECT id, nome, ruolo, squadra, quotazione_attuale, fvm
        FROM {NS}.giocatori
        WHERE lower(nome) LIKE lower(?)
        ORDER BY nome
        LIMIT 20
    """, [f"%{q}%"])
    return {"risultati": rows}


# ---------------------------------------------------------------------------
# SCHEDA GIOCATORE
# ---------------------------------------------------------------------------

@app.get("/giocatore/{giocatore_id}/scheda")
def scheda_giocatore(giocatore_id: int):
    rows = query_rows(f"""
        SELECT g.id, g.nome, g.ruolo, g.squadra,
               g.quotazione_attuale, g.quotazione_iniziale, g.fvm,
               mr.prezzo_pagato,
               CASE WHEN mr.giocatore_id IS NOT NULL THEN true ELSE false END AS in_rosa
        FROM {NS}.giocatori g
        LEFT JOIN {NS}.mia_rosa mr ON mr.giocatore_id = g.id
        WHERE g.id = ?
    """, [giocatore_id])
    if not rows:
        raise HTTPException(status_code=404, detail="Giocatore non trovato")
    g = rows[0]

    stagioni = query_rows(f"""
        SELECT stagione, presenze, media_voto, fantamedia,
               gol, assist, ammonizioni, espulsioni,
               gol_subiti, rigori_parati, rigori_calciati,
               bonus, malus, autogol
        FROM {NS}.statistiche_storiche
        WHERE giocatore_id = ?
        ORDER BY stagione ASC
    """, [giocatore_id])

    voti = query_rows(f"""
        SELECT giornata, voto, fantavoto, gol, assist,
               ammonizione, espulsione, stagione
        FROM {NS}.voti_giornata
        WHERE giocatore_id = ?
        ORDER BY stagione DESC, giornata ASC
        LIMIT 38
    """, [giocatore_id])

    trend_fantamedia = {
        "labels": [s["stagione"] for s in stagioni],
        "values": [float(s["fantamedia"]) if s["fantamedia"] is not None else None for s in stagioni],
    }

    def somma(campo):
        return sum((s[campo] or 0) for s in stagioni)

    radar = {
        "labels": ["Gol", "Assist", "Rigori parati", "Rigori calciati", "Ammonizioni", "Espulsioni", "Autogol"],
        "values": [
            somma("gol"), somma("assist"), somma("rigori_parati"),
            somma("rigori_calciati"), somma("ammonizioni"),
            somma("espulsioni"), somma("autogol"),
        ],
    }

    return {
        "giocatore":        g,
        "stagioni":         stagioni,
        "trend_fantamedia": trend_fantamedia,
        "radar_bonus_malus": radar,
        "voti_giornata":    voti,
    }


# ---------------------------------------------------------------------------
# CONFRONTO GIOCATORI
# ---------------------------------------------------------------------------

@app.get("/confronta")
def confronta(ids: str = Query(...)):
    try:
        id_list = [int(i.strip()) for i in ids.split(",")][:4]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids deve essere una lista di interi separati da virgola")

    # Spark SQL non ha ANY() — usiamo IN con valori inline (lista controllata, no injection)
    ids_sql = ",".join(str(i) for i in id_list)
    giocatori = query_rows(f"""
        SELECT g.id, g.nome, g.ruolo, g.squadra,
               g.quotazione_attuale, g.fvm,
               ROUND(AVG(s.fantamedia), 2)   AS fantamedia_media,
               ROUND(AVG(s.media_voto), 2)   AS media_voto_media,
               ROUND(AVG(s.presenze), 1)     AS presenze_media,
               ROUND(AVG(s.gol), 1)          AS gol_media,
               ROUND(AVG(s.assist), 1)       AS assist_media,
               ROUND(AVG(s.ammonizioni), 1)  AS ammonizioni_media,
               COUNT(s.stagione)             AS stagioni_disponibili
        FROM {NS}.giocatori g
        LEFT JOIN {NS}.statistiche_storiche s ON s.giocatore_id = g.id
        WHERE g.id IN ({ids_sql})
        GROUP BY g.id, g.nome, g.ruolo, g.squadra, g.quotazione_attuale, g.fvm
        ORDER BY fantamedia_media DESC NULLS LAST
    """)

    chart_voti = {
        "labels": ["Fantamedia", "Media voto"],
        "datasets": [
            {
                "label": f"{g['nome']} ({g['squadra']})",
                "data": [
                    float(g["fantamedia_media"] or 0),
                    float(g["media_voto_media"] or 0),
                ],
            }
            for g in giocatori
        ],
    }

    chart_presenze = {
        "labels": ["Presenze"],
        "datasets": [
            {
                "label": f"{g['nome']} ({g['squadra']})",
                "data": [float(g["presenze_media"] or 0)],
            }
            for g in giocatori
        ],
    }

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
        "giocatori":      giocatori,
        "chart_voti":     chart_voti,
        "chart_bonus":    chart_bonus,
        "chart_presenze": chart_presenze,
    }


# ---------------------------------------------------------------------------
# VALORE ATTESO
# ---------------------------------------------------------------------------

@app.get("/valore-atteso")
def valore_atteso(
    ruolo:      Optional[str] = Query(None),
    budget_max: Optional[int] = Query(None),
    limit:      int           = Query(20, le=50),
):
    sql = f"""
        SELECT g.id, g.nome, g.ruolo, g.squadra,
               g.quotazione_attuale, g.fvm,
               s.stagione, s.fantamedia, s.presenze
        FROM {NS}.giocatori g
        LEFT JOIN (
            SELECT giocatore_id, stagione, fantamedia, presenze,
                   ROW_NUMBER() OVER (
                       PARTITION BY giocatore_id ORDER BY stagione DESC
                   ) AS rn
            FROM {NS}.statistiche_storiche
        ) s ON s.giocatore_id = g.id AND s.rn <= 4
        WHERE g.id NOT IN (
            SELECT giocatore_id FROM {NS}.asta_log WHERE giocatore_id IS NOT NULL
        )
    """
    params = []
    if ruolo:
        sql += " AND g.ruolo = ?"
        params.append(ruolo)
    if budget_max:
        sql += " AND g.fvm <= ?"
        params.append(budget_max)
    sql += " ORDER BY g.id, s.stagione DESC"
    rows = query_rows(sql, params)

    PESI = [3, 2, 1, 0.5]

    def affidabilita(p: float) -> float:
        if p < 10: return 0.30
        if p < 20: return 0.60
        if p < 28: return 0.85
        return 1.00

    giocatori_map: dict = {}
    for r in rows:
        gid = r["id"]
        if gid not in giocatori_map:
            giocatori_map[gid] = {
                "id": r["id"], "nome": r["nome"], "ruolo": r["ruolo"],
                "squadra": r["squadra"],
                "quotazione_attuale": float(r["quotazione_attuale"] or 0),
                "fvm": float(r["fvm"] or 0),
                "stagioni": [],
            }
        if r["stagione"] is not None:
            giocatori_map[gid]["stagioni"].append({
                "stagione":   r["stagione"],
                "fantamedia": float(r["fantamedia"] or 0),
                "presenze":   float(r["presenze"]   or 0),
            })

    risultati = []
    for g in giocatori_map.values():
        stagioni    = g["stagioni"]
        n           = len(stagioni)
        pesi_usati  = PESI[:n]
        somma_pesi  = sum(pesi_usati)

        if n > 0 and somma_pesi > 0:
            fm_pond   = sum(stagioni[i]["fantamedia"] * pesi_usati[i] for i in range(n)) / somma_pesi
            pres_pond = sum(stagioni[i]["presenze"]   * pesi_usati[i] for i in range(n)) / somma_pesi
        else:
            fm_pond   = None
            pres_pond = 0

        risultati.append({
            **{k: v for k, v in g.items() if k != "stagioni"},
            "fantamedia_ponderata":  round(fm_pond, 2) if fm_pond is not None else None,
            "presenze_medie_pond":   round(pres_pond, 1),
            "affidabilita":          affidabilita(pres_pond),
            "n_stagioni":            n,
            "stagioni_dettaglio":    stagioni,
            "score":                 None,
        })

    for ruolo_corrente in set(g["ruolo"] for g in risultati):
        subset  = [g for g in risultati if g["ruolo"] == ruolo_corrente]
        max_fvm = max((g["fvm"] or 0) for g in subset) or 1
        max_fm  = max((g["fantamedia_ponderata"] or 0) for g in subset) or 1
        for g in subset:
            fvm_norm = (g["fvm"] or 0) / max_fvm
            fm_norm  = (g["fantamedia_ponderata"] or 0) / max_fm
            g["score"] = round((fvm_norm * 0.60 + fm_norm * 0.40) * g["affidabilita"] * 100, 1)

    risultati.sort(key=lambda g: g["score"] or 0, reverse=True)
    return {"giocatori": risultati[:limit]}


# ---------------------------------------------------------------------------
# TENDENZA MERCATO
# ---------------------------------------------------------------------------

@app.get("/tendenza-mercato")
def tendenza_mercato():
    ruoli = query_rows(f"""
        SELECT g.ruolo,
               COUNT(*)                                      AS acquisti,
               ROUND(AVG(a.prezzo_finale), 1)               AS prezzo_medio_pagato,
               ROUND(AVG(g.fvm), 1)                         AS fvm_medio,
               ROUND(AVG(stat.fantamedia_media), 2)         AS fantamedia_storica_media,
               ROUND(AVG(a.prezzo_finale) / NULLIF(AVG(g.fvm), 0), 2) AS rapporto_prezzo_fvm
        FROM {NS}.asta_log a
        JOIN {NS}.giocatori g ON g.id = a.giocatore_id
        LEFT JOIN (
            SELECT giocatore_id, AVG(fantamedia) AS fantamedia_media
            FROM {NS}.statistiche_storiche
            GROUP BY giocatore_id
        ) stat ON stat.giocatore_id = g.id
        GROUP BY g.ruolo
        ORDER BY g.ruolo
    """)

    etichette_ruolo = {"P": "Portieri", "D": "Difensori", "C": "Centrocampisti", "A": "Attaccanti"}
    risultati = []
    for r in ruoli:
        prezzo  = float(r["prezzo_medio_pagato"] or 0)
        fvm     = float(r["fvm_medio"] or 0)
        rapporto = round(prezzo / fvm, 2) if fvm > 0 else None
        if rapporto is None:
            valutazione, colore = "Dati insufficienti", "gray"
        elif rapporto > 1.15:
            valutazione = f"Il mercato sta pagando {round((rapporto-1)*100)}% sopra il valore atteso — ruolo gonfiato"
            colore = "red"
        elif rapporto < 0.85:
            valutazione = f"Il mercato sta pagando {round((1-rapporto)*100)}% sotto il valore atteso — buone occasioni"
            colore = "green"
        else:
            valutazione, colore = "Prezzi in linea con il valore atteso", "amber"
        risultati.append({
            **dict(r),
            "fvm_medio_norm":           fvm,
            "rapporto_prezzo_fvm_norm": rapporto,
            "valutazione":              valutazione,
            "colore":                   colore,
            "etichetta_ruolo":          etichette_ruolo.get(r["ruolo"], r["ruolo"]),
        })

    chart_prezzi = {
        "labels": [r["etichetta_ruolo"] for r in risultati],
        "datasets": [
            {"label": "Prezzo medio pagato",
             "data":  [float(r["prezzo_medio_pagato"] or 0) for r in risultati]},
            {"label": f"FVM medio (scala {BUDGET_LEGA})",
             "data":  [r["fvm_medio_norm"] for r in risultati]},
        ],
    }
    chart_rapporto = {
        "labels": [r["etichetta_ruolo"] for r in risultati],
        "datasets": [{"label": "Rapporto prezzo / FVM (1.0 = in linea)",
                      "data":  [r["rapporto_prezzo_fvm_norm"] for r in risultati]}],
        "soglia": 1.0,
    }

    return {
        "budget_lega":   BUDGET_LEGA,
        "per_ruolo":     risultati,
        "chart_prezzi":  chart_prezzi,
        "chart_rapporto": chart_rapporto,
    }


# ---------------------------------------------------------------------------
# RENDIMENTO MIA ROSA
# ---------------------------------------------------------------------------

@app.get("/mia-rosa/rendimento")
def rendimento_rosa():
    rosa = query_rows(f"""
        SELECT g.id, g.nome, g.ruolo, g.squadra, mr.prezzo_pagato,
               ROUND(AVG(s.fantamedia), 2) AS fantamedia_storica
        FROM {NS}.mia_rosa mr
        JOIN {NS}.giocatori g ON g.id = mr.giocatore_id
        LEFT JOIN {NS}.statistiche_storiche s ON s.giocatore_id = g.id
        GROUP BY g.id, g.nome, g.ruolo, g.squadra, mr.prezzo_pagato
        ORDER BY g.ruolo, fantamedia_storica DESC NULLS LAST
    """)

    voti_rows = []
    if rosa:
        ids_rosa_sql = ",".join(str(r["id"]) for r in rosa)
        voti_rows = query_rows(f"""
            SELECT giocatore_id, giornata, fantavoto
            FROM {NS}.voti_giornata
            WHERE giocatore_id IN ({ids_rosa_sql})
            ORDER BY giornata ASC
        """)

    voti_per_giocatore: dict = {}
    for v in voti_rows:
        gid = v["giocatore_id"]
        voti_per_giocatore.setdefault(gid, []).append({
            "giornata": v["giornata"],
            "fantavoto": float(v["fantavoto"]) if v["fantavoto"] else None,
        })

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
        "rosa": [{**r, "voti_giornata": voti_per_giocatore.get(r["id"], [])} for r in rosa],
        "chart_trend": chart_trend,
    }


# ---------------------------------------------------------------------------
# CLASSIFICA LEGA
# ---------------------------------------------------------------------------

@app.get("/classifica-lega")
def classifica_lega():
    classifica = query_rows(f"""
        SELECT a.squadra_acquirente              AS squadra,
               COUNT(DISTINCT vg.giornata)       AS giornate_giocate,
               ROUND(SUM(vg.fantavoto), 1)       AS punti_totali,
               ROUND(AVG(vg.fantavoto), 2)       AS fantamedia_squadra
        FROM {NS}.asta_log a
        JOIN {NS}.voti_giornata vg ON vg.giocatore_id = a.giocatore_id
        GROUP BY a.squadra_acquirente
        ORDER BY punti_totali DESC NULLS LAST
    """)

    chart = {
        "labels": [r["squadra"] for r in classifica],
        "datasets": [{"label": "Punti totali",
                      "data":  [float(r["punti_totali"] or 0) for r in classifica]}],
    }
    return {"classifica": classifica, "chart": chart}


# ---------------------------------------------------------------------------
# TOP/FLOP GIORNATA
# ---------------------------------------------------------------------------

@app.get("/giornata/{numero}/top")
def top_giornata(numero: int, limit: int = Query(10, le=20)):
    tutti = query_rows(f"""
        SELECT g.nome, g.ruolo, g.squadra,
               vg.voto, vg.fantavoto,
               vg.gol, vg.assist, vg.ammonizione, vg.espulsione,
               a.squadra_acquirente AS squadra_fanta
        FROM {NS}.voti_giornata vg
        JOIN {NS}.giocatori g ON g.id = vg.giocatore_id
        LEFT JOIN {NS}.asta_log a ON a.giocatore_id = vg.giocatore_id
        WHERE vg.giornata = ?
        ORDER BY vg.fantavoto DESC NULLS LAST
    """, [numero])

    top  = tutti[:limit]
    flop = sorted(tutti, key=lambda x: (x["fantavoto"] or 0))[:limit]
    return {"giornata": numero, "top": top, "flop": flop}

"""
FantAssistant - Import statistiche storiche da xlsx Fantacalcio.it

Formato atteso (stesso per tutte le stagioni dal 2015/16):
    Riga 1: titolo (es. "Statistiche Fantacalcio Stagione 2024 25")
    Riga 2: header colonne
    Colonne: Id, R, Rm, Nome, Squadra, Pv, Mv, Fm, Gf, Gs, Rp, Rc, R+, R-, Ass, Amm, Esp, Au

Uso (dentro al container etl):
    # Singolo file, stagione ricavata automaticamente dal titolo del foglio
    python import_statistiche.py /app/data/Statistiche_2024_25.xlsx

    # Più file in una volta (ultimi 4 campionati)
    python import_statistiche.py /app/data/Statistiche_2024_25.xlsx \\
                                 /app/data/Statistiche_2023_24.xlsx \\
                                 /app/data/Statistiche_2022_23.xlsx \\
                                 /app/data/Statistiche_2021_22.xlsx

    # Stagione esplicita (override, utile se il titolo manca o è strano)
    python import_statistiche.py /app/data/stats.xlsx --stagione 2024/25

Idempotente: fa upsert su (giocatore_id, stagione).
Rilanciarlo non crea duplicati, aggiorna solo i valori.

Join su fanta_id (colonna Id del file): stabile anche se il giocatore
cambia squadra tra stagioni. I giocatori non presenti in anagrafica
(non importati dal listone) vengono saltati con un warning.
"""
import os
import re
import sys

import pandas as pd
import psycopg

DATABASE_URL = os.environ["DATABASE_URL"]


def _ricava_stagione(xlsx_path: str, df_raw: pd.DataFrame) -> str:
    """
    Ricava la stringa stagione (es. '2024/25') dal titolo nella prima
    riga del foglio. Fallback: nome del file.
    """
    # Prova dal titolo nella cella A1 (es. "Statistiche Fantacalcio Stagione 2024 25")
    titolo = str(df_raw.iloc[0, 0]) if not df_raw.empty else ""
    m = re.search(r"(\d{4})\D+(\d{2,4})", titolo)
    if m:
        anno_inizio = m.group(1)
        anno_fine = m.group(2)[-2:]  # prendi sempre le ultime 2 cifre
        return f"{anno_inizio}/{anno_fine}"

    # Fallback: cerca nel nome del file
    m2 = re.search(r"(\d{4})[_\-\s](\d{2,4})", os.path.basename(xlsx_path))
    if m2:
        anno_inizio = m2.group(1)
        anno_fine = m2.group(2)[-2:]
        return f"{anno_inizio}/{anno_fine}"

    raise ValueError(
        f"Impossibile ricavare la stagione da '{xlsx_path}'. "
        "Passa --stagione YYYY/YY esplicitamente."
    )


def _to_int(val, default=0) -> int:
    try:
        if pd.isna(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _to_float(val) -> float | None:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def import_statistiche(xlsx_path: str, stagione_override: str | None = None):
    # Leggi raw per estrarre il titolo dalla riga 1
    df_raw = pd.read_excel(xlsx_path, sheet_name="Tutti", header=None)
    stagione = stagione_override or _ricava_stagione(xlsx_path, df_raw)

    # Leggi con header reale (riga 2, indice 1)
    df = pd.read_excel(xlsx_path, sheet_name="Tutti", header=1)

    colonne_richieste = {"Id", "Pv", "Mv", "Fm", "Gf", "Gs", "Rp", "Rc", "R+", "R-", "Ass", "Amm", "Esp", "Au"}
    mancanti = colonne_richieste - set(df.columns)
    if mancanti:
        raise ValueError(f"Colonne mancanti in '{xlsx_path}': {mancanti}")

    importati = saltati = aggiornati = 0
    incompleti = 0

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                if pd.isna(row.get("Id")) or pd.isna(row.get("Pv")):
                    incompleti += 1
                    continue  # riga vuota o malformata

                fanta_id = int(row["Id"])

                # Risolvi giocatore_id tramite fanta_id
                cur.execute(
                    "SELECT id FROM giocatori WHERE fanta_id = %s",
                    (fanta_id,),
                )
                result = cur.fetchone()
                if result is None:
                    saltati += 1
                    continue  # giocatore non in anagrafica, salta
                giocatore_id = result[0]

                cur.execute(
                    """
                    INSERT INTO statistiche_storiche (
                        giocatore_id, stagione,
                        presenze, media_voto, fantamedia,
                        gol, gol_subiti, rigori_parati, rigori_calciati,
                        bonus, malus, assist, ammonizioni, espulsioni, autogol
                    ) VALUES (
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (giocatore_id, stagione) DO UPDATE SET
                        presenze         = EXCLUDED.presenze,
                        media_voto       = EXCLUDED.media_voto,
                        fantamedia       = EXCLUDED.fantamedia,
                        gol              = EXCLUDED.gol,
                        gol_subiti       = EXCLUDED.gol_subiti,
                        rigori_parati    = EXCLUDED.rigori_parati,
                        rigori_calciati  = EXCLUDED.rigori_calciati,
                        bonus            = EXCLUDED.bonus,
                        malus            = EXCLUDED.malus,
                        assist           = EXCLUDED.assist,
                        ammonizioni      = EXCLUDED.ammonizioni,
                        espulsioni       = EXCLUDED.espulsioni,
                        autogol          = EXCLUDED.autogol
                    """,
                    (
                        giocatore_id, stagione,
                        _to_int(row["Pv"]),
                        _to_float(row["Mv"]),
                        _to_float(row["Fm"]),
                        _to_int(row["Gf"]),
                        _to_int(row["Gs"]),
                        _to_int(row["Rp"]),
                        _to_int(row["Rc"]),
                        _to_int(row["R+"]),
                        _to_int(row["R-"]),
                        _to_int(row["Ass"]),
                        _to_int(row["Amm"]),
                        _to_int(row["Esp"]),
                        _to_int(row["Au"]),
                    ),
                )

                # statusmessage è "INSERT 0 1" per nuovi record, "UPDATE 1" per aggiornamenti
                if cur.statusmessage and cur.statusmessage.startswith("INSERT"):
                    importati += 1
                else:
                    aggiornati += 1

            conn.commit()

    print(
        f"Stagione {stagione}: {importati} inseriti, {aggiornati} aggiornati, "
        f"{saltati} saltati (non in anagrafica), {incompleti} scartati (dati incompleti)."
    )
    return stagione, importati, aggiornati, saltati, incompleti


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(
            "Uso: python import_statistiche.py <file.xlsx> [file2.xlsx ...] [--stagione YYYY/YY]\n"
            "Esempio: python import_statistiche.py stats_24_25.xlsx stats_23_24.xlsx"
        )
        sys.exit(1)

    stagione_override = None
    file_paths = []
    i = 0
    while i < len(args):
        if args[i] == "--stagione" and i + 1 < len(args):
            stagione_override = args[i + 1]
            i += 2
        else:
            file_paths.append(args[i])
            i += 1

    if not file_paths:
        print("Errore: nessun file xlsx specificato.")
        sys.exit(1)

    if len(file_paths) > 1 and stagione_override:
        print(
            "Attenzione: --stagione ignorato quando si passano più file "
            "(la stagione viene ricavata da ciascun file automaticamente)."
        )
        stagione_override = None

    totale_importati = totale_aggiornati = totale_saltati = totale_incompleti = 0
    for path in file_paths:
        try:
            _, imp, agg, sal, inc = import_statistiche(path, stagione_override)
            totale_importati += imp
            totale_aggiornati += agg
            totale_saltati += sal
            totale_incompleti += inc
        except Exception as e:
            print(f"Errore su '{path}': {e}")
            sys.exit(1)

    if len(file_paths) > 1:
        print(
            f"\nTotale: {totale_importati} inseriti, {totale_aggiornati} aggiornati, "
            f"{totale_saltati} saltati (non in anagrafica), {totale_incompleti} scartati (dati incompleti) "
            f"su {len(file_paths)} file."
        )

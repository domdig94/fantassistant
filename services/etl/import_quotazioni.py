"""
FantAssistant - Import listone ufficiale Fantacalcio (xlsx)

Pensato per il formato di export standard (fogli: Tutti, Portieri,
Difensori, Centrocampisti, Attaccanti, Ceduti), con l'header vero alla
seconda riga del foglio (la prima riga e' un titolo).

Colonne attese nel foglio "Tutti":
    Id, R, RM, Nome, Squadra, Qt.A, Qt.I, Diff., Qt.A M, Qt.I M, Diff.M, FVM, FVM M

Uso (dentro al container etl):
    python import_quotazioni.py /app/data/Quotazioni_Fantacalcio.xlsx
    # Con rimozione automatica dei giocatori non piu' nel listone (fantasmi):
    python import_quotazioni.py /app/data/Quotazioni_Fantacalcio.xlsx --sync

Idempotente: fa upsert su (nome, squadra), quindi puoi rilanciarlo dopo
un aggiornamento del listone (es. dopo il calciomercato estivo) senza
creare duplicati - aggiorna solo i valori.
"""
import os
import sys

import pandas as pd
import psycopg

DATABASE_URL = os.environ["DATABASE_URL"]


def import_listone(xlsx_path: str, foglio: str = "Tutti", sync: bool = False):
    df = pd.read_excel(xlsx_path, sheet_name=foglio, header=1)

    richieste = {"Id", "R", "Nome", "Squadra", "Qt.A", "Qt.I", "FVM"}
    mancanti = richieste - set(df.columns)
    if mancanti:
        raise ValueError(f"Colonne mancanti nel file: {mancanti}")

    ids_nel_file = set(
        int(row) for row in df["Id"].dropna()
    )

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            count = 0
            for _, row in df.iterrows():
                # Riga vuota o malformata (capita a volte a fine foglio): salta
                if pd.isna(row["Nome"]) or pd.isna(row["Squadra"]):
                    continue

                cur.execute(
                    """
                    INSERT INTO giocatori (fanta_id, nome, ruolo, squadra, quotazione_iniziale, quotazione_attuale, fvm)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (fanta_id) DO UPDATE
                    SET nome = EXCLUDED.nome,
                        ruolo = EXCLUDED.ruolo,
                        squadra = EXCLUDED.squadra,          -- <--- Aggiorna la squadra se è cambiata!
                        quotazione_iniziale = EXCLUDED.quotazione_iniziale,
                        quotazione_attuale = EXCLUDED.quotazione_attuale,
                        fvm = EXCLUDED.fvm,
                        aggiornato_il = now()
                    """,
                    (
                        int(row["Id"]),
                        str(row["Nome"]).strip(),
                        str(row["R"]).strip(),
                        str(row["Squadra"]).strip(),
                        int(row["Qt.I"]),
                        int(row["Qt.A"]),
                        int(row["FVM"]),
                    ),
                )
                count += 1

            if sync:
                # Trova e rimuove i giocatori non piu' presenti nel listone.
                # ON DELETE CASCADE su statistiche_storiche e le altre tabelle
                # collegate garantisce che non restino righe orfane.
                cur.execute(
                    "SELECT id, fanta_id, nome, squadra FROM giocatori "
                    "WHERE fanta_id IS NOT NULL AND fanta_id != ALL(%s)",
                    (list(ids_nel_file),),
                )
                fantasmi = cur.fetchall()
                if fantasmi:
                    cur.execute(
                        "DELETE FROM giocatori WHERE fanta_id != ALL(%s) "
                        "AND fanta_id IS NOT NULL",
                        (list(ids_nel_file),),
                    )
                    print(f"Rimossi {len(fantasmi)} giocatori non piu' nel listone:")
                    for g in fantasmi:
                        print(f"  - {g[2]} ({g[3]}, fanta_id={g[1]})")
                else:
                    print("Nessun fantasma trovato.")

            conn.commit()

    print(f"Importati/aggiornati {count} giocatori dal foglio '{foglio}'.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Uso: python import_quotazioni.py <path_xlsx> [nome_foglio] [--sync]")
        sys.exit(1)

    sync = "--sync" in args
    args = [a for a in args if a != "--sync"]

    foglio = args[1] if len(args) > 1 else "Tutti"
    import_listone(args[0], foglio, sync=sync)

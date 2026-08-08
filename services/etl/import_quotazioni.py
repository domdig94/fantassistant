"""
FantAssistant - Import listone ufficiale Fantacalcio (xlsx)

Pensato per il formato di export standard (fogli: Tutti, Portieri,
Difensori, Centrocampisti, Attaccanti, Ceduti), con l'header vero alla
seconda riga del foglio (la prima riga e' un titolo).

Colonne attese nel foglio "Tutti":
    Id, R, RM, Nome, Squadra, Qt.A, Qt.I, Diff., Qt.A M, Qt.I M, Diff.M, FVM, FVM M

Uso (dentro al container etl):
    python import_quotazioni.py /app/data/Quotazioni_Fantacalcio.xlsx

Idempotente: fa upsert su (nome, squadra), quindi puoi rilanciarlo dopo
un aggiornamento del listone (es. dopo il calciomercato estivo) senza
creare duplicati - aggiorna solo i valori.
"""
import os
import sys

import pandas as pd
import psycopg

DATABASE_URL = os.environ["DATABASE_URL"]


def import_listone(xlsx_path: str, foglio: str = "Tutti"):
    df = pd.read_excel(xlsx_path, sheet_name=foglio, header=1)

    richieste = {"Id", "R", "Nome", "Squadra", "Qt.A", "Qt.I", "FVM"}
    mancanti = richieste - set(df.columns)
    if mancanti:
        raise ValueError(f"Colonne mancanti nel file: {mancanti}")

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
                    ON CONFLICT (nome, squadra) DO UPDATE
                    SET fanta_id = EXCLUDED.fanta_id,
                        ruolo = EXCLUDED.ruolo,
                        quotazione_iniziale = EXCLUDED.quotazione_iniziale,
                        quotazione_attuale = EXCLUDED.quotazione_attuale,
                        fvm = EXCLUDED.fvm
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
            conn.commit()

    print(f"Importati/aggiornati {count} giocatori dal foglio '{foglio}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python import_quotazioni.py <path_xlsx> [nome_foglio]")
        sys.exit(1)
    foglio = sys.argv[2] if len(sys.argv) > 2 else "Tutti"
    import_listone(sys.argv[1], foglio)

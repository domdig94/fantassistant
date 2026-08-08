"""
FantAssistant - ETL scheletro

Per iniziare senza vision e senza scraping: carica un CSV di quotazioni
(scaricabile manualmente da fonti pubbliche, es. il classico export
quotazioni Fantacalcio) dentro alla tabella `giocatori`.

Uso (dentro al container etl):
    python import_csv.py /app/data/quotazioni.csv

Formato CSV atteso (colonne, header incluso):
    nome,ruolo,squadra,quotazione

Estendi questo script quando vuoi aggiungere altre fonti (statistiche
storiche, calendario, voti giornata) seguendo lo stesso pattern.
"""
import os
import sys
import csv

import psycopg

DATABASE_URL = os.environ["DATABASE_URL"]


def import_giocatori(csv_path: str):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    cur.execute(
                        """
                        INSERT INTO giocatori (nome, ruolo, squadra, quotazione_iniziale, quotazione_attuale)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (nome, squadra) DO UPDATE
                        SET quotazione_attuale = EXCLUDED.quotazione_attuale
                        """,
                        (
                            row["nome"],
                            row["ruolo"],
                            row["squadra"],
                            row["quotazione"],
                            row["quotazione"],
                        ),
                    )
                    count += 1
            conn.commit()
    print(f"Importati/aggiornati {count} giocatori.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python import_csv.py <path_csv>")
        sys.exit(1)
    import_giocatori(sys.argv[1])

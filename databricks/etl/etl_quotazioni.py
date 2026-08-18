"""
FantAssistant Databricks — ETL listone quotazioni
Equivalente di services/etl/import_quotazioni.py, adattato per Delta Lake.

Può girare come:
  - Databricks Job (Task tipo Python script, cluster con DBR 14+)
  - Notebook (incollare il contenuto in celle)

Input:  file xlsx caricato su DBFS o Unity Catalog Volume
        es. /Volumes/fantassistant/main/uploads/listone.xlsx
Output: upsert su Delta Table fantassistant.main.giocatori

Parametri (passati come job parameters o env vars):
  XLSX_PATH      path DBFS/Volume del file xlsx
  FOGLIO         nome foglio (default: Tutti)
  SYNC           'true' per rimuovere i giocatori non piu' nel listone
  BUDGET_LEGA    budget lega (default: 1000)
  UNITY_CATALOG  catalog Unity (default: fantassistant)
  UNITY_SCHEMA   schema Unity  (default: main)
"""
import math
import os
import sys

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    LongType, StringType, DoubleType
)
from delta.tables import DeltaTable

spark = SparkSession.builder.getOrCreate()

CATALOG     = os.environ.get("UNITY_CATALOG", "fantassistant")
SCHEMA      = os.environ.get("UNITY_SCHEMA",  "main")
XLSX_PATH   = os.environ.get("XLSX_PATH",     "/Volumes/fantassistant/main/uploads/listone.xlsx")
FOGLIO      = os.environ.get("FOGLIO",        "Tutti")
SYNC        = os.environ.get("SYNC",          "false").lower() == "true"
BUDGET_LEGA = int(os.environ.get("BUDGET_LEGA", "1000"))

TABLE = f"`{CATALOG}`.`{SCHEMA}`.giocatori"


def _build_testo_embedding(nome, ruolo, squadra, fvm, quot_att):
    """
    Testo libero usato come sorgente per il Vector Search embedding.
    Mantieni aggiornato se aggiungi campi rilevanti.
    """
    ruolo_label = {"P": "Portiere", "D": "Difensore", "C": "Centrocampista", "A": "Attaccante"}.get(ruolo, ruolo)
    return (
        f"{nome} è un {ruolo_label} del {squadra}. "
        f"Quotazione attuale: {quot_att} crediti. "
        f"Fantavalore di mercato (FVM): {fvm} crediti."
    )


def import_listone(xlsx_path: str, foglio: str = "Tutti", sync: bool = False, budget_lega: int = 1000):
    fvm_scale = budget_lega / 1000
    if fvm_scale != 1.0:
        print(f"FVM scalato da base 1000 a budget {budget_lega} (fattore {fvm_scale})")

    df = pd.read_excel(xlsx_path, sheet_name=foglio, header=1)
    richieste = {"Id", "R", "Nome", "Squadra", "Qt.A", "Qt.I", "FVM"}
    mancanti = richieste - set(df.columns)
    if mancanti:
        raise ValueError(f"Colonne mancanti nel file: {mancanti}")

    rows = []
    ids_nel_file = set()
    for _, row in df.iterrows():
        if pd.isna(row["Nome"]) or pd.isna(row["Squadra"]):
            continue
        fanta_id  = int(row["Id"])
        nome      = str(row["Nome"]).strip()
        ruolo     = str(row["R"]).strip()
        squadra   = str(row["Squadra"]).strip()
        quot_i    = float(row["Qt.I"])
        quot_a    = float(row["Qt.A"])
        fvm       = float(math.ceil(float(row["FVM"]) * fvm_scale))
        testo     = _build_testo_embedding(nome, ruolo, squadra, fvm, quot_a)
        ids_nel_file.add(fanta_id)
        rows.append((fanta_id, nome, ruolo, squadra, quot_i, quot_a, fvm, testo))

    schema = StructType([
        StructField("fanta_id",            LongType(),   True),
        StructField("nome",                StringType(), False),
        StructField("ruolo",               StringType(), False),
        StructField("squadra",             StringType(), False),
        StructField("quotazione_iniziale", DoubleType(), True),
        StructField("quotazione_attuale",  DoubleType(), True),
        StructField("fvm",                 DoubleType(), True),
        StructField("testo_embedding",     StringType(), True),
    ])
    incoming = spark.createDataFrame(rows, schema=schema)

    # ---------- UPSERT su Delta Table ----------
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.giocatori")
    (
        dt.alias("target")
        .merge(incoming.alias("src"), "target.fanta_id = src.fanta_id")
        .whenMatchedUpdate(set={
            "nome":                "src.nome",
            "ruolo":               "src.ruolo",
            "squadra":             "src.squadra",
            "quotazione_iniziale": "src.quotazione_iniziale",
            "quotazione_attuale":  "src.quotazione_attuale",
            "fvm":                 "src.fvm",
            "testo_embedding":     "src.testo_embedding",
            "aggiornato_il":       F.current_timestamp(),
        })
        .whenNotMatchedInsert(values={
            "fanta_id":            "src.fanta_id",
            "nome":                "src.nome",
            "ruolo":               "src.ruolo",
            "squadra":             "src.squadra",
            "quotazione_iniziale": "src.quotazione_iniziale",
            "quotazione_attuale":  "src.quotazione_attuale",
            "fvm":                 "src.fvm",
            "testo_embedding":     "src.testo_embedding",
            "creato_il":           F.current_timestamp(),
            "aggiornato_il":       F.current_timestamp(),
        })
        .execute()
    )
    print(f"Upsert completato: {len(rows)} giocatori dal foglio '{foglio}'.")

    # ---------- SYNC: rimuovi fantasmi ----------
    if sync:
        ids_list = list(ids_nel_file)
        fantasmi = spark.sql(f"""
            SELECT id, fanta_id, nome, squadra
            FROM {TABLE}
            WHERE fanta_id IS NOT NULL
              AND fanta_id NOT IN ({','.join(str(x) for x in ids_list)})
        """)
        n_fantasmi = fantasmi.count()
        if n_fantasmi > 0:
            for r in fantasmi.collect():
                print(f"  Rimosso: {r['nome']} ({r['squadra']}, fanta_id={r['fanta_id']})")
            spark.sql(f"""
                DELETE FROM {TABLE}
                WHERE fanta_id IS NOT NULL
                  AND fanta_id NOT IN ({','.join(str(x) for x in ids_list)})
            """)
            print(f"Rimossi {n_fantasmi} giocatori non piu' nel listone.")
        else:
            print("Nessun fantasma trovato.")


if __name__ == "__main__":
    import_listone(XLSX_PATH, FOGLIO, sync=SYNC, budget_lega=BUDGET_LEGA)

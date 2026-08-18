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
from pyspark.sql.types import (
    StructType, StructField,
    LongType, StringType, DoubleType
)
from upsert_utils import upsert_giocatori

spark = SparkSession.builder.getOrCreate()

# Parsing parametri: accetta KEY=VALUE da command line (Job parameters)
# oppure da environment variables, con fallback ai default.
_cli_args = {}
for arg in sys.argv[1:]:
    if "=" in arg:
        k, v = arg.split("=", 1)
        _cli_args[k] = v

def _param(name: str, default: str) -> str:
    return _cli_args.get(name, os.environ.get(name, default))

CATALOG     = _param("UNITY_CATALOG", "platform")
SCHEMA      = _param("UNITY_SCHEMA",  "fantassistant")
XLSX_PATH   = _param("XLSX_PATH",     "/Volumes/platform/fantassistant/uploads/listone.xlsx")
FOGLIO      = _param("FOGLIO",        "Tutti")
SYNC        = _param("SYNC",          "false").lower() == "true"
BUDGET_LEGA = int(_param("BUDGET_LEGA", "1000"))

TABLE = f"`{CATALOG}`.`{SCHEMA}`.giocatori"



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
        ids_nel_file.add(fanta_id)
        rows.append((fanta_id, nome, ruolo, squadra, quot_i, quot_a, fvm))

    schema = StructType([
        StructField("fanta_id",            LongType(),   True),
        StructField("nome",                StringType(), False),
        StructField("ruolo",               StringType(), False),
        StructField("squadra",             StringType(), False),
        StructField("quotazione_iniziale", DoubleType(), True),
        StructField("quotazione_attuale",  DoubleType(), True),
        StructField("fvm",                 DoubleType(), True),
    ])
    incoming = spark.createDataFrame(rows, schema=schema)

    # ---------- UPSERT su Delta Table ----------
    upsert_giocatori(incoming)
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

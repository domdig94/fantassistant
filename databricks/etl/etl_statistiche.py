"""
FantAssistant Databricks — ETL statistiche storiche
Equivalente di services/etl/import_statistiche.py, adattato per Delta Lake.

Parametri (env vars o job parameters):
  XLSX_PATHS     path separati da virgola (es. /Volumes/.../stat_2425.xlsx,/Volumes/.../stat_2324.xlsx)
  STAGIONE       override stagione (es. 2024/25) — opzionale
  UNITY_CATALOG  catalog Unity (default: fantassistant)
  UNITY_SCHEMA   schema Unity  (default: main)
"""
import os
import re

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, StringType, DoubleType,
)
from delta.tables import DeltaTable

spark = SparkSession.builder.getOrCreate()

CATALOG           = os.environ.get("UNITY_CATALOG", "fantassistant")
SCHEMA            = os.environ.get("UNITY_SCHEMA",  "main")
XLSX_PATHS_RAW    = os.environ.get("XLSX_PATHS",    "")
STAGIONE_OVERRIDE = os.environ.get("STAGIONE",      None)


def _ricava_stagione(xlsx_path: str, df_raw: pd.DataFrame) -> str:
    titolo = str(df_raw.iloc[0, 0]) if not df_raw.empty else ""
    m = re.search(r"(\d{4})\D+(\d{2,4})", titolo)
    if m:
        return f"{m.group(1)}/{m.group(2)[-2:]}"
    m2 = re.search(r"(\d{4})[_\-\s](\d{2,4})", os.path.basename(xlsx_path))
    if m2:
        return f"{m2.group(1)}/{m2.group(2)[-2:]}"
    raise ValueError(
        f"Impossibile ricavare la stagione da '{xlsx_path}'. Passa STAGIONE=YYYY/YY."
    )


def _to_int(val, default=0) -> int:
    try:
        return default if pd.isna(val) else int(val)
    except (TypeError, ValueError):
        return default


def _to_float(val):
    try:
        return None if pd.isna(val) else float(val)
    except (TypeError, ValueError):
        return None


def import_statistiche(xlsx_path: str, stagione_override: str | None = None):
    df_raw = pd.read_excel(xlsx_path, sheet_name="Tutti", header=None)
    stagione = stagione_override or _ricava_stagione(xlsx_path, df_raw)
    df = pd.read_excel(xlsx_path, sheet_name="Tutti", header=1)

    colonne_richieste = {"Id", "Pv", "Mv", "Fm", "Gf", "Gs", "Rp", "Rc", "R+", "R-", "Ass", "Amm", "Esp", "Au"}
    mancanti = colonne_richieste - set(df.columns)
    if mancanti:
        raise ValueError(f"Colonne mancanti in '{xlsx_path}': {mancanti}")

    # Recupera mapping fanta_id -> id da Delta Table giocatori
    giocatori_df = spark.sql(
        f"SELECT id, fanta_id FROM `{CATALOG}`.`{SCHEMA}`.giocatori WHERE fanta_id IS NOT NULL"
    )
    fanta_id_map = {r["fanta_id"]: r["id"] for r in giocatori_df.collect()}

    rows = []
    saltati = incompleti = 0
    for _, row in df.iterrows():
        if pd.isna(row.get("Id")) or pd.isna(row.get("Pv")):
            incompleti += 1
            continue
        fanta_id = int(row["Id"])
        giocatore_id = fanta_id_map.get(fanta_id)
        if giocatore_id is None:
            saltati += 1
            continue
        rows.append((
            giocatore_id,
            stagione,
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
        ))

    schema = StructType([
        StructField("giocatore_id",    LongType(),    False),
        StructField("stagione",        StringType(),  False),
        StructField("presenze",        IntegerType(), True),
        StructField("media_voto",      DoubleType(),  True),
        StructField("fantamedia",      DoubleType(),  True),
        StructField("gol",             IntegerType(), True),
        StructField("gol_subiti",      IntegerType(), True),
        StructField("rigori_parati",   IntegerType(), True),
        StructField("rigori_calciati", IntegerType(), True),
        StructField("bonus",           IntegerType(), True),
        StructField("malus",           IntegerType(), True),
        StructField("assist",          IntegerType(), True),
        StructField("ammonizioni",     IntegerType(), True),
        StructField("espulsioni",      IntegerType(), True),
        StructField("autogol",         IntegerType(), True),
    ])
    incoming = spark.createDataFrame(rows, schema=schema)

    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.statistiche_storiche")
    (
        dt.alias("target")
        .merge(
            incoming.alias("src"),
            "target.giocatore_id = src.giocatore_id AND target.stagione = src.stagione"
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(
        f"Stagione {stagione}: {len(rows)} upsertati, "
        f"{saltati} saltati (non in anagrafica), {incompleti} scartati."
    )


if __name__ == "__main__":
    paths = [p.strip() for p in XLSX_PATHS_RAW.split(",") if p.strip()]
    if not paths:
        print("Imposta XLSX_PATHS con i path dei file xlsx separati da virgola.")
    for path in paths:
        import_statistiche(path, stagione_override=STAGIONE_OVERRIDE)

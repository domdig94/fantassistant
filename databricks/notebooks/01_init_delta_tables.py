# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# FantAssistant — Creazione tabelle Delta Lake in Unity Catalog
# Eseguire UNA SOLA VOLTA (o in caso di reset completo)
# Le tabelle sono equivalenti allo schema PostgreSQL in db/init.sql

from databricks.sdk.runtime import spark  # disponibile nei notebook Databricks

CATALOG = "platform"
SCHEMA  = "fantassistant"
NS      = f"`{CATALOG}`.`{SCHEMA}`"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {NS}")

# --------------------------------------------------------------------------
# giocatori
# --------------------------------------------------------------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NS}.giocatori (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fanta_id         INT,
    nome             STRING NOT NULL,
    ruolo            STRING NOT NULL,
    squadra          STRING NOT NULL,
    quotazione_iniziale DOUBLE,
    quotazione_attuale  DOUBLE,
    fvm              DOUBLE,
    creato_il        TIMESTAMP DEFAULT current_timestamp(),
    aggiornato_il    TIMESTAMP DEFAULT current_timestamp()
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true', 'delta.feature.allowColumnDefaults' = 'supported')
""")

# --------------------------------------------------------------------------
# statistiche_storiche
# --------------------------------------------------------------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NS}.statistiche_storiche (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    giocatore_id     BIGINT NOT NULL,
    stagione         STRING NOT NULL,
    presenze         INT    DEFAULT 0,
    gol              INT    DEFAULT 0,
    assist           INT    DEFAULT 0,
    media_voto       DOUBLE,
    fantamedia       DOUBLE,
    gol_subiti       INT    DEFAULT 0,
    rigori_parati    INT    DEFAULT 0,
    rigori_calciati  INT    DEFAULT 0,
    bonus            INT    DEFAULT 0,
    malus            INT    DEFAULT 0,
    autogol          INT    DEFAULT 0,
    ammonizioni      INT    DEFAULT 0,
    espulsioni       INT    DEFAULT 0,
    CONSTRAINT stat_storiche_fk FOREIGN KEY (giocatore_id) REFERENCES {NS}.giocatori(id)
)

USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
""")

# --------------------------------------------------------------------------
# squadre
# --------------------------------------------------------------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NS}.squadre (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nome           STRING NOT NULL,
    allenatore     STRING,
    budget_totale  DOUBLE NOT NULL,
    creato_il      TIMESTAMP DEFAULT current_timestamp()
)
USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
""")

# --------------------------------------------------------------------------
# asta_log
# --------------------------------------------------------------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NS}.asta_log (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    giocatore_id       BIGINT,
    prezzo_finale      DOUBLE,
    squadra_acquirente STRING,
    fonte              STRING,
    creato_il          TIMESTAMP DEFAULT current_timestamp(),
    CONSTRAINT asta_log_fk FOREIGN KEY (giocatore_id) REFERENCES {NS}.giocatori(id)
)
USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
""")

# --------------------------------------------------------------------------
# mia_rosa
# --------------------------------------------------------------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NS}.mia_rosa (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    giocatore_id  BIGINT,
    prezzo_pagato DOUBLE,
    ruolo_fanta   STRING,
    acquistato_il TIMESTAMP DEFAULT current_timestamp(),
    CONSTRAINT mia_rosa_fk FOREIGN KEY (giocatore_id) REFERENCES {NS}.giocatori(id)
)
USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
""")

# --------------------------------------------------------------------------
# voti_giornata
# --------------------------------------------------------------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NS}.voti_giornata (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    giocatore_id BIGINT,
    giornata     INT    NOT NULL,
    stagione     STRING NOT NULL,
    voto         DOUBLE,
    fantavoto    DOUBLE,
    gol          INT    DEFAULT 0,
    assist       INT    DEFAULT 0,
    ammonizione  BOOLEAN DEFAULT FALSE,
    espulsione   BOOLEAN DEFAULT FALSE,
    CONSTRAINT voti_fk FOREIGN KEY (giocatore_id) REFERENCES {NS}.giocatori(id)
)

USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
""")

# --------------------------------------------------------------------------
# calendario
# --------------------------------------------------------------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NS}.calendario (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    giornata          INT    NOT NULL,
    squadra_casa      STRING NOT NULL,
    squadra_trasferta STRING NOT NULL,
    data_partita      TIMESTAMP,
    stagione          STRING NOT NULL
)
USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
""")

# --------------------------------------------------------------------------
# formazioni_probabili
# --------------------------------------------------------------------------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NS}.formazioni_probabili (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    giornata   INT    NOT NULL,
    squadra    STRING NOT NULL,
    modulo     STRING,
    titolari   ARRAY<STRING>,
    fonte      STRING,
    creato_il  TIMESTAMP DEFAULT current_timestamp()
)
USING DELTA
TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
""")

print(f"Tabelle create in {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Aggiunta CHECK constraints
# --------------------------------------------------------------------------
# CHECK constraints (Delta Lake li supporta solo via ALTER TABLE)
# --------------------------------------------------------------------------

constraints = [
    # giocatori
    ("giocatori", "ruolo_check", "ruolo IN ('P','D','C','A')"),
    ("giocatori", "quotazione_iniziale_pos", "quotazione_iniziale >= 0"),
    ("giocatori", "quotazione_attuale_pos", "quotazione_attuale >= 0"),
    # mia_rosa
    ("mia_rosa", "ruolo_fanta_check", "ruolo_fanta IN ('P','D','C','A')"),
    ("mia_rosa", "prezzo_pagato_pos", "prezzo_pagato >= 0"),
    # statistiche_storiche
    ("statistiche_storiche", "presenze_pos", "presenze >= 0"),
    ("statistiche_storiche", "gol_pos", "gol >= 0"),
    ("statistiche_storiche", "assist_pos", "assist >= 0"),
    # voti_giornata
    ("voti_giornata", "giornata_range", "giornata BETWEEN 1 AND 38"),
    ("voti_giornata", "voti_gol_pos", "gol >= 0"),
    ("voti_giornata", "voti_assist_pos", "assist >= 0"),
    # squadre
    ("squadre", "budget_pos", "budget_totale > 0"),
    # asta_log
    ("asta_log", "prezzo_finale_pos", "prezzo_finale >= 0"),
    # calendario
    ("calendario", "cal_giornata_range", "giornata BETWEEN 1 AND 38"),
    # formazioni_probabili
    ("formazioni_probabili", "form_giornata_range", "giornata BETWEEN 1 AND 38"),
]

for table, name, expr in constraints:
    try:
        spark.sql(f"ALTER TABLE {NS}.{table} ADD CONSTRAINT {name} CHECK ({expr})")
        print(f"  + {table}.{name}")
    except Exception as e:
        if "CONSTRAINT_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
            print(f"  ~ {table}.{name} (già presente)")
        else:
            print(f"  ! {table}.{name} ERRORE: {e}")

print("\nCHECK constraints completati.")
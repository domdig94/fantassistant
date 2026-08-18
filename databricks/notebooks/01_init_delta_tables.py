# Databricks notebook source
# FantAssistant — Creazione tabelle Delta Lake in Unity Catalog
# Eseguire UNA SOLA VOLTA (o in caso di reset completo)
# Le tabelle sono equivalenti allo schema PostgreSQL in db/init.sql

import os
from databricks.sdk.runtime import spark  # disponibile nei notebook Databricks

CATALOG = os.environ.get("UNITY_CATALOG", "fantassistant")
SCHEMA  = os.environ.get("UNITY_SCHEMA",  "main")
NS      = f"`{CATALOG}`.`{SCHEMA}`"

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

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
    aggiornato_il    TIMESTAMP DEFAULT current_timestamp(),
    CONSTRAINT ruolo_check CHECK (ruolo IN ('P','D','C','A')),
    CONSTRAINT giocatori_fanta_id_unique UNIQUE (fanta_id),
    CONSTRAINT giocatori_nome_squadra_unique UNIQUE (nome, squadra)
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
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
    CONSTRAINT stat_storiche_fk FOREIGN KEY (giocatore_id) REFERENCES {NS}.giocatori(id),
    CONSTRAINT stat_storiche_unique UNIQUE (giocatore_id, stagione)
)
USING DELTA
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
    creato_il      TIMESTAMP DEFAULT current_timestamp(),
    CONSTRAINT squadre_nome_unique UNIQUE (nome)
)
USING DELTA
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
    CONSTRAINT voti_fk FOREIGN KEY (giocatore_id) REFERENCES {NS}.giocatori(id),
    CONSTRAINT voti_unique UNIQUE (giocatore_id, giornata, stagione)
)
USING DELTA
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
""")

print(f"Tabelle create in {CATALOG}.{SCHEMA}")

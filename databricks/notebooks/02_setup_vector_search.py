# Databricks notebook source
# FantAssistant — Creazione Vector Search endpoint + indice
# Prerequisito: tabella giocatori già popolata con Change Data Feed attivo
# Eseguire dopo il primo ETL quotazioni.

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    VectorIndexType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
)

w = WorkspaceClient()

CATALOG  = os.environ.get("UNITY_CATALOG", "fantassistant")
SCHEMA   = os.environ.get("UNITY_SCHEMA",  "main")
VS_ENDPOINT = os.environ.get("VECTOR_SEARCH_ENDPOINT", "fantassistant-vs")

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.giocatori"
INDEX_NAME   = f"{CATALOG}.{SCHEMA}.giocatori_vs_index"

# --------------------------------------------------------------------------
# 1. Crea endpoint Vector Search (se non esiste)
# --------------------------------------------------------------------------
existing = [e.name for e in w.vector_search_endpoints.list_endpoints()]
if VS_ENDPOINT not in existing:
    print(f"Creo endpoint '{VS_ENDPOINT}'...")
    w.vector_search_endpoints.create_endpoint_and_wait(
        name=VS_ENDPOINT,
        endpoint_type=EndpointType.STANDARD,
    )
    print("Endpoint creato.")
else:
    print(f"Endpoint '{VS_ENDPOINT}' già esistente.")

# --------------------------------------------------------------------------
# 2. Crea indice Delta Sync (si aggiorna automaticamente al cambio tabella)
# --------------------------------------------------------------------------
# La colonna 'testo_embedding' va creata nella tabella giocatori come colonna
# calcolata o aggiornata dall'ETL. Vedi etl/etl_quotazioni.py per la logica.
# L'embedding model è gestito internamente da Databricks Vector Search
# (non serve un deployment Azure OAI separato per gli embedding).
# --------------------------------------------------------------------------

try:
    print(f"Creo indice '{INDEX_NAME}'...")
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=VS_ENDPOINT,
        primary_key="id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=SOURCE_TABLE,
            pipeline_type=PipelineType.TRIGGERED,   # cambia in CONTINUOUS se vuoi real-time
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="testo_embedding",           # colonna testuale da vettorializzare
                    embedding_model_endpoint_name="databricks-gte-large-en",  # modello built-in
                )
            ],
        ),
    )
    print("Indice creato.")
except Exception as e:
    if "already exists" in str(e).lower():
        print("Indice già esistente, nessuna azione.")
    else:
        raise

print(f"""
Setup Vector Search completato.
  Endpoint : {VS_ENDPOINT}
  Indice   : {INDEX_NAME}
  Tabella  : {SOURCE_TABLE}

Ricorda di impostare la variabile VECTOR_SEARCH_ENDPOINT={VS_ENDPOINT}
nel secret scope 'fantassistant' prima di avviare le Databricks Apps.
""")

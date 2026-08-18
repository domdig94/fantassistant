# FantAssistant — Bozza migrazione Databricks

Questa cartella contiene la versione **Databricks-native** di FantAssistant.
Il codice originale in `services/` rimane invariato e continua a funzionare su Podman.

## Architettura target

```
Databricks Workspace
├── Apps
│   ├── fantassistant-analytics     (FastAPI)
│   ├── fantassistant-chatbot       (FastAPI)
│   ├── fantassistant-vision        (FastAPI)
│   ├── fantassistant-auction       (FastAPI)
│   └── fantassistant-frontend      (HTML statico)
├── Workflows
│   ├── etl_quotazioni              (Job Python)
│   └── etl_statistiche             (Job Python)
└── Unity Catalog
    └── catalog: fantassistant
        └── schema: main
            ├── giocatori           (Delta Table)
            ├── statistiche_storiche(Delta Table)
            ├── asta_log            (Delta Table)
            ├── mia_rosa            (Delta Table)
            ├── squadre             (Delta Table)
            ├── voti_giornata       (Delta Table)
            ├── calendario          (Delta Table)
            └── formazioni_probabili(Delta Table)
```

## Sostituzioni chiave rispetto alla versione Podman

| Componente originale | Equivalente Databricks |
|---|---|
| PostgreSQL | Delta Lake + Unity Catalog |
| psycopg / SQL diretto | `databricks-sql-connector` oppure `spark.sql()` nei Job |
| Chroma + ONNXMiniLM | Databricks Vector Search (Mosaic) |
| pgvector | Databricks Vector Search |
| Container Podman | Databricks Apps |
| ETL script Python | Databricks Workflows (Job Python) |
| `.env` secrets | Databricks Secrets (secret scope) |

## Ordine di deploy consigliato

1. Creare il catalog e lo schema in Unity Catalog
2. Eseguire `notebooks/01_init_delta_tables.py` per creare le tabelle Delta
3. Eseguire i Job ETL (`etl/etl_quotazioni.py`, `etl/etl_statistiche.py`)
4. Creare il Vector Search endpoint e l'indice (`notebooks/02_setup_vector_search.py`)
5. Deployare le Databricks Apps dalla cartella `apps/`

## Secrets richiesti (Databricks Secret Scope: `fantassistant`)

```
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION
AZURE_CHAT_DEPLOYMENT
DATABRICKS_HOST          (auto-disponibile dentro le App)
DATABRICKS_TOKEN         (auto-disponibile dentro le App)
UNITY_CATALOG            (es. fantassistant)
UNITY_SCHEMA             (es. main)
VECTOR_SEARCH_ENDPOINT   (nome endpoint VS creato al punto 4)
MY_TEAM
BUDGET_LEGA
```

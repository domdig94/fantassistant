# FantAssistant

Assistente per il Fantacalcio: asta, formazioni, statistiche, chatbot.
Container orchestrati con podman, AI via Azure OpenAI (GPT-4.1 per chat e
vision). Il retrieval del chatbot usa Chroma come servizio dedicato in
locale (con modello di embedding scaricato automaticamente), niente
deployment di embedding su Azure.

## Struttura

```
fantassistant/
├── podman-compose.yml
├── .env.example
├── db/init.sql                 # schema Postgres + pgvector
├── services/
│   ├── chatbot/                # RAG: FastAPI + Ollama + Postgres/pgvector
│   ├── vision/                 # OCR intelligente: legge screenshot/foto -> JSON
│   ├── auction-tracker/        # logica pura budget/asta, nessuna AI
│   └── etl/                    # import dati (parti da un CSV, poi estendi)
└── frontend/                   # (da fare) webapp o CLI unificata
```

## Prerequisiti

- podman + podman-compose (`sudo dnf install podman podman-compose` su RHEL/Fedora,
  o equivalente sulla tua distro)
- Una risorsa Azure OpenAI attiva, con un deployment di `gpt-4.1` (usato sia
  per chat che per vision, essendo multimodale)

  Nota: nei campi `AZURE_*_DEPLOYMENT` del `.env` vanno i **nomi che hai dato
  tu ai deployment**, non necessariamente uguali al nome del modello.

## Setup iniziale

Estrai lo zip, poi dentro alla cartella `fantassistant/`:

```bash
cp .env.example .env
```

Apri `.env` con un editor e compila:
- `AZURE_OPENAI_ENDPOINT` (es. `https://tuo-resource.openai.azure.com`)
- `AZURE_OPENAI_API_KEY`
- `AZURE_CHAT_DEPLOYMENT` e `AZURE_VISION_DEPLOYMENT` (puoi mettere lo
  stesso nome in entrambi, dato che GPT-4.1 gestisce sia testo che immagini)

Poi avvia tutto:

```bash
podman-compose up -d --build
```

Servizi esposti:
- chatbot:         http://localhost:8000
- vision:          http://localhost:8001
- auction-tracker: http://localhost:8002
- chroma:          http://localhost:8003
- postgres:        localhost:5432

## Import del listone ufficiale (xlsx)

Se hai il file ufficiale delle quotazioni (formato con fogli Tutti/Portieri/
Difensori/Centrocampisti/Attaccanti/Ceduti, header alla seconda riga):

```bash
podman-compose up -d --build etl

podman cp Quotazioni_Fantacalcio.xlsx fanta-etl:/app/listone.xlsx
podman exec -it fanta-etl python import_quotazioni.py /app/listone.xlsx
```

Importa dal foglio "Tutti" per default (tutti i ruoli insieme, 400-500
giocatori). Lo script fa upsert su (nome, squadra): rilanciarlo dopo un
aggiornamento del listone (es. dopo il calciomercato) non crea duplicati,
aggiorna solo i valori.

Poi ricostruisci l'indice RAG e fai una domanda vera:

```bash
curl -X POST http://localhost:8000/ingest

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"domanda": "Chi sono i centrocampisti con FVM piu alto?"}'
```

## Test rapido con pochi dati (CSV a mano)

Se invece vuoi solo verificare che il motore funzioni con pochi dati finti,
senza usare il listone reale:

1. Prepara un CSV `nome,ruolo,squadra,quotazione`
2. Copialo nel container etl e importalo:

```bash
podman cp quotazioni.csv fanta-etl:/app/quotazioni.csv
podman exec -it fanta-etl python import_csv.py /app/quotazioni.csv
```

3. `/ingest` e `/chat` come sopra.

## Vision: prova rapida

```bash
curl -X POST http://localhost:8001/parse/listone \
  -F "file=@/percorso/screenshot_listone.png"
```

Ritorna sia la risposta grezza (`raw_response`) sia il JSON estratto
(`parsed`). Con GPT-4.1 e il JSON mode di Azure il parsing e' molto piu'
affidabile rispetto a un modello locale piccolo, ma su foto di scarsa
qualita' o layout complessi puo' comunque sbagliare qualche voce: conviene
sempre fare un controllo/conferma umana prima di scrivere su Postgres,
specialmente per l'asta live dove un errore costa caro.

## Auction tracker: prova rapida

```bash
# imposta budget iniziale per ruolo
curl -X POST http://localhost:8002/budget -H "Content-Type: application/json" \
  -d '{"ruolo": "A", "budget_totale": 200}'

# registra un acquisto (tuo)
curl -X POST http://localhost:8002/acquisto -H "Content-Type: application/json" \
  -d '{"giocatore_id": 1, "prezzo_finale": 35, "squadra_acquirente": "Io", "e_mio": true}'

# controlla budget residuo
curl http://localhost:8002/budget
```

## Roadmap suggerita

1. [x] Scaffold: compose, schema, chatbot RAG base, vision base, auction tracker
2. [ ] Popolare dati storici reali (statistiche_storiche, calendario)
3. [ ] Affinare i prompt vision con screenshot reali (ogni fonte ha un layout diverso)
4. [ ] Automatizzare ETL (schedulazione periodica invece di lancio manuale)
5. [ ] Servizio `analytics`: calcolo "valore atteso" giocatore, suggerimenti asta
6. [ ] Frontend semplice (anche solo una CLI first, poi eventualmente webapp)

## Note tecniche

- Chroma gira come **servizio dedicato** (`fanta-chroma`, immagine ufficiale
  `chromadb/chroma`), raggiunto dal chatbot via HTTP client
  (`CHROMA_HOST`/`CHROMA_PORT`). Dati persistiti sul volume podman
  `chroma_data`, sopravvivono a restart/rebuild dei container.
- Al primo `/ingest` (o alla prima query) Chroma scarica il modello di
  embedding di default (~90MB, gira su CPU via ONNX) e lo tiene in cache
  nel container: la primissima chiamata sara' piu' lenta delle successive.
- Il client nel servizio `chatbot` inizializza la collection in modo
  "lazy" (alla prima richiesta, non all'avvio del container): evita che
  `chatbot` vada in crash-loop se per qualche motivo si avvia prima che
  Chroma sia pronto, nonostante l'healthcheck in compose dovrebbe gia'
  gestire l'ordine di avvio.
- `/ingest` fa un `upsert` (aggiorna se l'id giocatore esiste gia', crea
  altrimenti): richiamarlo piu' volte non duplica i documenti.
- Il modello di embedding di default di Chroma (`all-MiniLM-L6-v2`) e'
  multilingua ma allenato prevalentemente su inglese: per query semplici
  su nomi/ruoli/squadre funziona bene, su frasi italiane piu' complesse
  puo' essere un po' meno preciso di un embedding OpenAI. Se in futuro
  vuoi migliorarlo, Chroma supporta funzioni di embedding custom (anche
  Azure OpenAI, se un giorno avrai un deployment disponibile).
- I servizi montano il codice come volume (`:Z` per SELinux su RHEL) cosi'
  puoi modificare e vedere i cambi a caldo con `--reload` senza rebuildare
  l'immagine ogni volta durante lo sviluppo.
- **Costi**: a differenza di Ollama, ogni chiamata a `/chat` e `/parse/*`
  consuma token a pagamento su Azure (`/ingest` e' completamente gratuito
  ora, gira tutto in locale). In fase di sviluppo/test puo' valere la pena
  limitare le chiamate o impostare budget/alert lato Azure.
- Le chiavi vanno nel `.env`, che e' pensato per restare **fuori da git**
  (aggiungi un `.gitignore` con `.env` appena inizializzi il repo, se non
  l'hai gia' fatto — nello scaffold c'e' gia').

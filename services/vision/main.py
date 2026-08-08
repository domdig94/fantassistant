"""
FantAssistant - Vision service (Azure OpenAI GPT-4.1)
Usa GPT-4.1 come modello multimodale per leggere screenshot/foto e
trasformarli in dati strutturati. Tre casi d'uso principali:

  POST /parse/listone    -> foto/screenshot del listone quotazioni -> lista giocatori
  POST /parse/formazione -> screenshot formazione -> modulo + titolari
  POST /parse/asta-board -> foto tabellone/lavagna asta -> chi ha preso chi

Usiamo il "JSON mode" di Azure OpenAI (response_format) cosi' non serve
fare parsing best-effort del testo libero come si farebbe con un modello
locale piccolo: la risposta e' gia' JSON valido.
"""
import os
import base64
import json

from fastapi import FastAPI, UploadFile, File
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)
VISION_DEPLOYMENT = os.environ.get("AZURE_VISION_DEPLOYMENT", "gpt-4.1")

app = FastAPI(title="FantAssistant Vision")


def ask_vision(image_bytes: bytes, mime_type: str, system_prompt: str) -> dict:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=VISION_DEPLOYMENT,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            },
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    try:
        return {"raw_response": raw, "parsed": json.loads(raw)}
    except json.JSONDecodeError:
        return {"raw_response": raw, "parsed": None}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse/listone")
async def parse_listone(file: UploadFile = File(...)):
    system_prompt = """Guarda l'immagine: e' un listone di quotazioni Fantacalcio.
Estrai TUTTI i giocatori visibili. Rispondi SOLO con un oggetto JSON con questa
forma esatta (chiave "giocatori" con un array):

{"giocatori": [{"nome": "...", "ruolo": "P|D|C|A", "squadra": "...", "quotazione": 0}]}"""
    content = await file.read()
    return ask_vision(content, file.content_type or "image/png", system_prompt)


@app.post("/parse/formazione")
async def parse_formazione(file: UploadFile = File(...)):
    system_prompt = """Guarda l'immagine: e' una formazione di calcio schierata (es. Fantacalcio).
Rispondi SOLO con un oggetto JSON con questa forma esatta:

{"modulo": "es. 3-4-3", "titolari": ["nome1", "nome2", "..."]}"""
    content = await file.read()
    return ask_vision(content, file.content_type or "image/png", system_prompt)


@app.post("/parse/asta-board")
async def parse_asta_board(file: UploadFile = File(...)):
    system_prompt = """Guarda l'immagine: e' un tabellone/lavagna/foglio di un'asta Fantacalcio,
con nomi di giocatori, prezzi pagati e chi li ha acquistati.
Rispondi SOLO con un oggetto JSON con questa forma esatta (chiave "acquisti"):

{"acquisti": [{"nome": "...", "prezzo_finale": 0, "squadra_acquirente": "..."}]}"""
    content = await file.read()
    return ask_vision(content, file.content_type or "image/png", system_prompt)

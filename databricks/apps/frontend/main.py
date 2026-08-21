"""
FantAssistant Databricks — Frontend App
Serve i file HTML statici tramite FastAPI StaticFiles.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="FantAssistant Frontend")

@app.get("/")
def index():
    return FileResponse("index.html")

@app.get("/analytics")
def analytics():
    return FileResponse("analytics.html")

app.mount("/", StaticFiles(directory="."), name="static")

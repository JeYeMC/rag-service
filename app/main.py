# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logger import logger
from app.api import ingest, query, analyze, feedback

app = FastAPI(
    title="CRM RAG Service",
    description="Sistema RAG inteligente para documentos de CRM.",
    version="2.0.0"
)

# ------------ CORS ------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Cambia esto en producción
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# ------------ Rutas ------------
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(analyze.router)
app.include_router(feedback.router)

# ------------ Healthcheck ------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "CRM RAG"}

logger.info("🔥 CRM RAG Service iniciado correctamente.")

import os
import shutil
import json
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

from rag_pipeline import ingest_file_to_pinecone, answer_question
from ingest_utils import extract_text

# ==========================
# CONFIGURACIÓN INICIAL
# ==========================
load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="CRM RAG Service",
    description="Servicio RAG inteligente para CRM: contratos, correos, facturas, PQRS y propuestas.",
    version="2.0.0"
)

# ==========================
# CORS (para conexión con backend .NET)
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes restringir a ["http://localhost:4200"] o tu dominio backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# MODELOS DE ENTRADA
# ==========================
class Question(BaseModel):
    query: str
    doc_type: str | None = None  # permite filtrar búsquedas específicas

class Feedback(BaseModel):
    question: str
    answer: str
    doc_type: str | None = None
    correct: bool
    comment: str | None = None

# ==========================
# ENDPOINTS PRINCIPALES
# ==========================
@app.get("/health")
async def health():
    return {"status": "ok", "message": "CRM RAG Service funcionando correctamente."}

# ---------- INGESTA DE DOCUMENTOS ----------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...), source_name: str = Form("upload")):
    """Sube e indexa un documento en Pinecone."""
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    result = ingest_file_to_pinecone(str(dest), source_name=source_name)
    return {"upload_status": result, "filename": file.filename}

# ---------- CONSULTAS RAG ----------
@app.post("/query")
async def query_rag(q: Question):
    """
    Endpoint para consultar el sistema RAG.
    - Detecta tipo de documento automáticamente si no se pasa 'doc_type'.
    - Devuelve respuesta razonada, fuentes y diagnóstico.
    """
    start = time.time()
    result = answer_question(q.query, top_k=10)
    elapsed = round(time.time() - start, 2)
    return {
        "query": q.query,
        "doc_type": q.doc_type,
        "response": result["answer"],
        "sources": result["sources"],
        "diagnostics": result["diagnostics"],
        "elapsed_seconds": elapsed
    }

# ---------- FEEDBACK (para mejorar modelo) ----------
FEEDBACK_LOG = Path("feedback_log.json")

@app.post("/feedback")
async def feedback(data: Feedback):
    """
    Guarda feedback sobre las respuestas generadas.
    Esto se puede usar luego para reentrenar prompts o mejorar embeddings.
    """
    record = data.dict()
    record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    previous = []
    if FEEDBACK_LOG.exists():
        try:
            previous = json.loads(FEEDBACK_LOG.read_text())
        except Exception:
            previous = []
    previous.append(record)
    FEEDBACK_LOG.write_text(json.dumps(previous, indent=2, ensure_ascii=False))
    return {"status": "feedback_saved", "count": len(previous)}

# ---------- ANÁLISIS DE DOCUMENTOS ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Analiza automáticamente un documento y devuelve resumen + tipo detectado.
    No ingesta, solo analiza.
    """
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    text = extract_text(str(dest))
    snippet = text[:1000] + ("..." if len(text) > 1000 else "")
    # detección rápida de tipo
    doc_type = "contrato" if "cláusula" in text.lower() else "correo" if "estimado" in text.lower() else "otro"

    return {
        "filename": file.filename,
        "detected_doc_type": doc_type,
        "text_preview": snippet
    }

# app/api/ingest.py

from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil
import time

from app.core.logger import logger
from app.rag.ingestion import ingest_file_to_pinecone

router = APIRouter(prefix="/ingest", tags=["Ingesta"])

UPLOAD_DIR = Path("storages/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/")
async def ingest_document(
    file: UploadFile = File(...),
    provider: str = Form("hf"),               # <--- NUEVO: HF o OpenAI
    source_name: str = Form("upload")
):
    """
    Sube un archivo y lo procesa:
      - extrae texto
      - detecta tipo
      - chunk + embeddings con HF u OpenAI
      - analiza imágenes si es PDF
      - sube a Pinecone
      - devuelve metadata para el backend .NET
    """

    start = time.time()

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Archivo recibido: {dest} usando proveedor '{provider}'")

    # 🔥 Pasamos el provider hacia la pipeline RAG
    result = ingest_file_to_pinecone(
        str(dest),
        source_name=source_name,
        provider=provider
    )

    elapsed = round(time.time() - start, 2)
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "filename": file.filename,
        "result": result
    }

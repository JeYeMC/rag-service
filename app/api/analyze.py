# app/api/analyze.py

from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from app.utils.text_extract import extract_text
from app.rag.ingestion import detect_document_type

router = APIRouter(prefix="/analyze", tags=["Análisis"])

ANALYZE_DIR = Path("storages/analyze")
ANALYZE_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/")
async def analyze_document(file: UploadFile = File(...)):
    """
    Devuelve:
      - preview texto
      - tipo de documento
      - tamaño
    """
    dest = ANALYZE_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text(str(dest))
    preview = text[:1200]
    doc_type = detect_document_type(text)

    return {
        "filename": file.filename,
        "length": len(text),
        "preview": preview,
        "doc_type": doc_type,
    }

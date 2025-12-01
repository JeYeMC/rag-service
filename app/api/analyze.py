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
      - preview texto (limpio)
      - tipo de documento
      - tamaño
    """
    dest = ANALYZE_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # extract_text devuelve dict
    result = extract_text(str(dest))

    cleaned = result.get("cleaned_text", "")
    raw = result.get("raw_text", "")

    # preview basado en texto limpio
    preview = cleaned[:1200] if cleaned else raw[:1200]

    # detectar tipo basado en cleaned text
    doc_type = detect_document_type(cleaned)

    return {
        "filename": file.filename,
        "length": len(cleaned),
        "preview": preview,
        "doc_type": doc_type,
    }

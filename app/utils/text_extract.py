# app/utils/text_extract.py
import re
from pathlib import Path
import pymupdf as fitz  # PyMuPDF
import docx
import mimetypes
from app.core.logger import logger


def extract_text(file_path: str) -> str:
    """
    Extrae texto desde PDF, DOCX, TXT.
    Usa PyMuPDF para PDF.
    No usa OCR (Tesseract eliminado).
    """
    path = Path(file_path)
    mime, _ = mimetypes.guess_type(path)

    try:
        # PDF
        if mime == "application/pdf" or path.suffix.lower() == ".pdf":
            logger.info(f"Extrayendo texto de PDF con PyMuPDF: {path}")
            return extract_text_pdf(path)

        # DOCX
        if path.suffix.lower() == ".docx":
            logger.info(f"Extrayendo texto de DOCX: {path}")
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        # TXT
        if mime == "text/plain" or path.suffix.lower() == ".txt":
            logger.info(f"Extrayendo texto de TXT: {path}")
            return path.read_text(encoding="utf-8", errors="ignore")

        # Imágenes → no OCR (se elimina Tesseract)
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff"]:
            logger.info(f"Imagen detectada pero OCR está deshabilitado: {path}")
            return "[Imagen detectada: OCR deshabilitado]"

        logger.warning(f"Tipo de archivo no soportado: {path.suffix}")
        return ""

    except Exception as e:
        logger.error(f"❌ Error extrayendo texto de {file_path}: {e}")
        return ""


# ======================================================================
# PDF TEXT EXTRACTOR (PyMuPDF)
# ======================================================================
def extract_text_pdf(path: Path) -> str:
    """
    Extrae todo el texto de un PDF usando PyMuPDF.
    """
    text = ""
    doc = fitz.open(path)

    try:
        for page in doc:
            text += page.get_text("text") + "\n"
    finally:
        doc.close()

    return clean_text(text)


# ======================================================================
# LIMPIEZA BÁSICA
# ======================================================================
def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

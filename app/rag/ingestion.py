# app/rag/ingestion.py
import os
import uuid
import time
from typing import Tuple
from app.core.config import settings
from app.core.logger import logger
from app.utils.text_extract import extract_text
from app.utils.chunker import chunk_text
from app.utils.pdf_utils import analyze_pdf_images
from app.rag.embeddings import embed_texts
from app.vectorstore.pinecone_client import create_index, upsert_vectors

# Detección simple de tipo (puedes reemplazar por ML si quieres)
DOC_PATTERNS = {
    "contrato": ["contrato", "contratante", "contratista", "cláusula", "clausula", "honorarios"],
    "correo": ["asunto:", "estimado", "saludos", "atentamente", "from:", "para:"],
    "factura": ["factura", "subtotal", "iva", "valor total", "nit", "número de factura"],
    "propuesta": ["propuesta", "cotización", "alcance", "entregables"],
    "pqr": ["petición", "queja", "reclamo", "pqrs"],
    "acta": ["acta", "reunión", "acuerdos", "asistentes", "orden del día"]
}

def detect_document_type(text: str) -> str:
    t = text.lower()
    scores = {}
    for k, kws in DOC_PATTERNS.items():
        scores[k] = sum(1 for kw in kws if kw in t)
    best = max(scores, key=scores.get)
    if scores[best] == 0 or list(scores.values()).count(scores[best]) > 1:
        return "documento"
    return best

def ingest_file_to_pinecone(file_path: str, source_name: str = "upload", chunk_size: int = 500) -> dict:
    """
    Ingesta completa:
      - extrae texto
      - detecta tipo
      - analiza imágenes (si PDF)
      - chunk -> embed -> upsert (Pinecone)
      - devuelve payload listo para que .NET cree la entidad Documento
    """

    logger.info(f"Iniciando ingesta: {file_path}")
    start_t = time.time()

    if not os.path.exists(file_path):
        return {"status": "error", "error": "file_not_found", "msg": f"No existe: {file_path}"}

    text = extract_text(file_path)
    if not text or not text.strip():
        return {"status": "error", "error": "no_text_extracted"}

    filename = os.path.basename(file_path)
    filesize = os.path.getsize(file_path)

    doc_type = detect_document_type(text)
    document_id = str(uuid.uuid4())

    # analizar imagenes si es PDF
    num_images = 0
    images_meta = []
    if filename.lower().endswith(".pdf"):
        try:
            num_images, images_meta = analyze_pdf_images(file_path)
        except Exception as e:
            logger.warning(f"Error al analizar imágenes PDF: {e}")

    # chunking
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.2))
    vectors = embed_texts(chunks)

    # upsert en pinecone (metadata por chunk)
    upserts = []
    for i, vec in enumerate(vectors):
        chunk_id = str(uuid.uuid4())
        metadata = {
            "source": source_name,
            "chunk_index": i,
            "document_id": document_id,
            "text_excerpt": chunks[i][:600],
            "doc_type": doc_type,
            "filename": filename
        }
        upserts.append((chunk_id, vec, metadata))

    create_index(settings.PINECONE_INDEX, dim=len(vectors[0]) if vectors else 1536)
    upsert_vectors(settings.PINECONE_INDEX, upserts)

    # Resumen breve (fallback: primeros 1000 chars; ideal: LLM)
    resumen = text[:1000]
    # (opcional) si tienes LLM configurado puedes llamar la función de resumen aquí

    payload = {
        "status": "ok",
        "filename": filename,
        "document_id": document_id,
        "doc_type": doc_type,
        "contenido_extraido": text[:5000],
        "resumen_documento": resumen,
        "tamaño_archivo": filesize,
        "numero_imagenes": num_images,
        "imagenes_metadata": images_meta,
        "archivo_metadata_json": {
            "document_id": document_id,
            "filename": filename,
            "doc_type": doc_type,
            "chunks": len(chunks),
            "vector_dim": len(vectors[0]) if vectors else None,
            "source": source_name,
            "numero_imagenes": num_images
        },
        "elapsed_seconds": round(time.time() - start_t, 2)
    }

    logger.info(f"Ingesta completada: {filename} -> {document_id} (chunks={len(chunks)}, images={num_images})")
    return payload

# ---------------------------
# Ejemplo de uso (archivo que subiste)
# /mnt/data/RETOS PROYECTO INTEGRADOR 2025-2 (1) (2).pdf
# payload = ingest_file_to_pinecone("/mnt/data/RETOS PROYECTO INTEGRADOR 2025-2 (1) (2).pdf")
# print(payload)
# ---------------------------

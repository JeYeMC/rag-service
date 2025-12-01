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
from app.rag.llm_router import generate_summary   # NUEVO

from app.vectorstore.pinecone_client import create_index, upsert_vectors

# ------------------------------
# Patrones de detección de tipo
# ------------------------------
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
    scores = {k: sum(1 for kw in kws if kw in t) for k, kws in DOC_PATTERNS.items()}
    best = max(scores, key=scores.get)

    if scores[best] == 0 or list(scores.values()).count(scores[best]) > 1:
        return "documento"
    return best


# ================================================================
# 🔥 INGESTA PRINCIPAL (AHORA CON SELECCIÓN DE PROVEEDOR)
# ================================================================
def ingest_file_to_pinecone(
    file_path: str,
    source_name: str = "upload",
    chunk_size: int = 500,
    provider: str = None,          # <--- NUEVO
) -> dict:

    """
    Ingesta completa con selección de proveedor:
      - 'hf'      → Embeddings HF + LLM HF
      - 'openai'  → Embeddings OpenAI + LLM OpenAI
      - 'local'   → SentenceTransformers + LLM según settings (HF u OpenAI)
    """

    logger.info(f"Iniciando ingesta [{provider}] : {file_path}")
    start_t = time.time()

    if not os.path.exists(file_path):
        return {"status": "error", "error": "file_not_found", "msg": f"No existe: {file_path}"}

    # ------------------------------
    # 1) EXTRAER TEXTO
    # ------------------------------
    text = extract_text(file_path)
    if not text.strip():
        return {"status": "error", "error": "no_text_extracted"}

    filename = os.path.basename(file_path)
    filesize = os.path.getsize(file_path)
    doc_type = detect_document_type(text)
    document_id = str(uuid.uuid4())

    # ------------------------------
    # 2) ANALIZAR IMÁGENES (PDF)
    # ------------------------------
    num_images = 0
    images_meta = []
    if filename.lower().endswith(".pdf"):
        try:
            num_images, images_meta = analyze_pdf_images(file_path)
        except Exception as e:
            logger.warning(f"Error analizando imágenes PDF: {e}")

    # ------------------------------
    # 3) CHUNKING
    # ------------------------------
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.20)
    )

    # ------------------------------
    # 4) EMBEDDINGS (dependiendo del provider)
    # ------------------------------
    vectors = embed_texts(
        chunks,
        provider=provider   # <--- NUEVO
    )

    # ------------------------------
    # 5) UPSERT EN PINECONE
    # ------------------------------
    upserts = []
    for i, vec in enumerate(vectors):
        chunk_id = str(uuid.uuid4())
        metadata = {
            "source": source_name,
            "chunk_index": i,
            "document_id": document_id,
            "text_excerpt": chunks[i][:600],
            "doc_type": doc_type,
            "filename": filename,
            "provider": provider     # <--- IMPORTANTE PARA SABER CÓMO RESPONDIO
        }
        upserts.append((chunk_id, vec, metadata))

    create_index(settings.PINECONE_INDEX, dim=len(vectors[0]))
    upsert_vectors(settings.PINECONE_INDEX, upserts)

    # ------------------------------
    # 6) RESUMEN (LLM DINÁMICO)
    # ------------------------------
    try:
        resumen = generate_summary(text, provider=provider)
    except Exception as e:
        logger.warning(f"Fallo resumen LLM: {e}")
        resumen = text[:1200]   # fallback

    # ------------------------------
    # 7) RESPUESTA
    # ------------------------------
    payload = {
        "status": "ok",
        "filename": filename,
        "document_id": document_id,
        "doc_type": doc_type,
        "contenido_extraido": text,
        "resumen_documento": resumen,
        "tamaño_archivo": filesize,
        "numero_imagenes": num_images,
        "imagenes_metadata": images_meta,
        "archivo_metadata_json": {
            "document_id": document_id,
            "filename": filename,
            "doc_type": doc_type,
            "chunks": len(chunks),
            "vector_dim": len(vectors[0]),
            "source": source_name,
            "provider": provider,
            "numero_imagenes": num_images
        },
        "elapsed_seconds": round(time.time() - start_t, 2)
    }

    logger.info(
        f"Ingesta completada [{provider}]: {filename} -> {document_id} "
        f"(chunks={len(chunks)}, images={num_images})"
    )

    return payload

# app/rag/retriever.py

from typing import List, Optional
from app.core.logger import logger
from app.core.config import settings

from app.rag.embeddings import embed_texts
from app.vectorstore.pinecone_client import query_index

from sentence_transformers import CrossEncoder

# -------------------------------------------
# Cross Encoders por provider (carga perezosa)
# -------------------------------------------
_cross_encoders = {}

def get_cross_encoder(provider: str = "hf"):
    """
    Devuelve un cross-encoder según el provider.
    Si falla, devuelve None (el pipeline usa fallback).
    """
    global _cross_encoders

    if provider in _cross_encoders:
        return _cross_encoders[provider]

    model_name = settings.CROSS_ENCODER_MODEL

    try:
        ce = CrossEncoder(model_name)
        _cross_encoders[provider] = ce
        logger.info(f"Cargado cross-encoder {model_name} para provider={provider}")
    except Exception as e:
        logger.warning(f"No se pudo cargar cross-encoder para {provider}: {e}")
        _cross_encoders[provider] = None

    return _cross_encoders[provider]


# =====================================================
# 1. RETRIEVE — soporta provider + filtrado por metadata
# =====================================================

def retrieve(
    query: str,
    top_k: int = 20,
    doc_type: Optional[str] = None,
    provider: Optional[str] = None
) -> List[dict]:
    """
    Recupera chunks desde Pinecone con:
    - provider (HF/OpenAI/local)
    - doc_type (email/contrato/etc)
    """

    # ----- Generar embedding con el proveedor correcto -----
    qvec = embed_texts([query], provider=provider)[0]

    # ----- Filtrado en Pinecone -----
    filter_obj = {}

    if provider:
        filter_obj["provider"] = {"$eq": provider}

    if doc_type:
        filter_obj["doc_type"] = {"$eq": doc_type}

    if not filter_obj:
        filter_obj = None

    # Buscar en un pool grande y luego seleccionar top_k
    pool_k = max(top_k * 4, 50)

    res = query_index(
        index_name=settings.PINECONE_INDEX,
        vector=qvec,
        top_k=pool_k,
        include_metadata=True,
        filter=filter_obj
    )

    matches = (
        res.get("matches", [])
        if isinstance(res, dict)
        else res.matches if hasattr(res, "matches")
        else []
    )

    hits = []
    for m in matches:
        meta = (
            m.get("metadata", {})
            if isinstance(m, dict)
            else getattr(m, "metadata", {}) or {}
        )

        hits.append({
            "id": m.get("id") if isinstance(m, dict) else getattr(m, "id", None),
            "score": m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0),
            "metadata": meta
        })

    # Ordenar por score bruto
    hits_sorted = sorted(hits, key=lambda x: x["score"], reverse=True)

    return hits_sorted[:top_k]


# =====================================================
# 2. RERANK — CrossEncoder dinámico por provider
# =====================================================

def rerank(
    query: str,
    hits: List[dict],
    top_k: int = 10,
    provider: Optional[str] = None
) -> List[dict]:

    if not hits:
        return []

    ce = get_cross_encoder(provider or "hf")

    if not ce:
        logger.debug("No cross-encoder disponible — devolviendo top-k directo.")
        return hits[:top_k]

    # Preparar pares (query, chunk)
    pairs = [(query, h["metadata"].get("text_excerpt", "")) for h in hits]

    try:
        scores = ce.predict(pairs)
    except Exception as e:
        logger.warning(f"Cross-encoder falló: {e}")
        return hits[:top_k]

    for i, h in enumerate(hits):
        h["_rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

    hits_reranked = sorted(hits, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)

    return hits_reranked[:top_k]

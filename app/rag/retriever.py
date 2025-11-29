# app/rag/retriever.py
from typing import List, Optional
from app.core.logger import logger
from app.core.config import settings
from app.rag.embeddings import embed_texts
from app.vectorstore.pinecone_client import query_index
from sentence_transformers import CrossEncoder

# Cross-encoder para re-ranking
_cross_encoder = None
def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            _cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)
            logger.info(f"Cargado cross-encoder: {settings.CROSS_ENCODER_MODEL}")
        except Exception as e:
            logger.warning(f"No se pudo cargar cross-encoder: {e}")
            _cross_encoder = None
    return _cross_encoder

def retrieve(query: str, top_k: int = 20, doc_type: Optional[str] = None) -> List[dict]:
    """
    Recupera hits desde Pinecone. Si doc_type se pasa, filtramos por metadata.
    """
    qvec = embed_texts([query])[0]
    pool_k = max(top_k * 4, 50)
    # construir filtro si aplica (Pinecone SDK usa operators)
    filter_obj = None
    if doc_type:
        filter_obj = {"doc_type": {"$eq": doc_type}}

    res = query_index(settings.PINECONE_INDEX, qvec, top_k=pool_k, include_metadata=True, filter=filter_obj)
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches if hasattr(res, "matches") else []

    hits = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        hits.append({
            "id": m.get("id") if isinstance(m, dict) else getattr(m, "id", None),
            "score": m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0),
            "metadata": meta
        })

    # ordenar por score (ya vienen ordenados, pero por si acaso)
    hits_sorted = sorted(hits, key=lambda x: x["score"], reverse=True)
    return hits_sorted[:top_k]


def rerank(query: str, hits: List[dict], top_k: int = 10) -> List[dict]:
    """
    Re-rank usando CrossEncoder si está disponible.
    """
    if not hits:
        return []

    ce = get_cross_encoder()
    if not ce:
        logger.debug("No cross-encoder disponible — devolviendo hits sin re-rank.")
        return hits[:top_k]

    pairs = [(query, h["metadata"].get("text_excerpt", "")) for h in hits]
    try:
        scores = ce.predict(pairs)
    except Exception as e:
        logger.warning(f"Cross-encoder fallo: {e}")
        return hits[:top_k]

    for i, h in enumerate(hits):
        h["_rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

    hits_reranked = sorted(hits, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
    return hits_reranked[:top_k]

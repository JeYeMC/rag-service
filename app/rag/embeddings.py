# app/rag/embeddings.py
from app.core.config import settings
from app.core.logger import logger

# Cargador de embeddings local (sentence-transformers) por defecto.
# Cambia la implementación si quieres usar OpenAI/HF en producción.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        if settings.EMB_PROVIDER == "sentence_transformers" and SentenceTransformer is not None:
            logger.info(f"Cargando modelo de embeddings: {settings.EMB_MODEL}")
            _model = SentenceTransformer(settings.EMB_MODEL)
        else:
            # Fallback: no model available
            logger.warning("No hay un proveedor de embeddings configurado o SentenceTransformer no está instalado.")
            _model = None
    return _model

def embed_texts(texts):
    """
    Recibe lista de textos y devuelve lista de vectores (list of lists).
    """
    model = get_embedding_model()
    if model is None:
        raise RuntimeError("Modelo de embeddings no disponible. Revisa settings.")
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()

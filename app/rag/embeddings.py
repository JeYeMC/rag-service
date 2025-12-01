# app/rag/embeddings.py

import requests
from app.core.config import settings
from app.core.logger import logger

# ============================
# Local sentence-transformers
# ============================
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_local_model = None


# ============================
# CARGA DEL MODELO LOCAL
# ============================
def _load_local_model():
    global _local_model
    if _local_model is None:
        if SentenceTransformer is None:
            logger.error("sentence-transformers no está instalado.")
            return None

        logger.info(f"🔹 Cargando modelo local ST: {settings.EMB_MODEL}")
        _local_model = SentenceTransformer(settings.EMB_MODEL)

    return _local_model


# ============================
# HUGGINGFACE INFERENCE API
# ============================
def _hf_embed(texts: list[str]) -> list[list[float]]:
    if settings.HF_INFERENCE_API_KEY is None or settings.HF_MODEL is None:
        raise RuntimeError("Faltan variables HF: HF_INFERENCE_API_KEY o HF_MODEL")

    url = settings.HF_API_URL or f"https://api-inference.huggingface.co/pipeline/feature-extraction/{settings.HF_MODEL}"

    headers = {"Authorization": f"Bearer {settings.HF_INFERENCE_API_KEY}"}

    logger.info(f"🔹 Usando HuggingFace Inference API para embeddings: {settings.HF_MODEL}")

    response = requests.post(url, headers=headers, json={"inputs": texts})

    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace embedding error: {response.text}")

    return response.json()


# ============================
# OPENAI EMBEDDINGS
# ============================
def _openai_embed(texts: list[str]) -> list[list[float]]:
    if settings.OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY no configurada.")

    import openai
    openai.api_key = settings.OPENAI_API_KEY

    logger.info("🔹 Usando OpenAI embeddings (text-embedding-3-large)")

    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )

    return [item.embedding for item in response.data]


# ============================
# INTERFAZ PRINCIPAL
# ============================
def embed_texts(texts: list[str], provider: str | None = None) -> list[list[float]]:
    """
    Devuelve lista de embeddings.
    provider puede ser:
        - "sentence_transformers"
        - "hf"
        - "openai"
        - None → usa EMB_PROVIDER del .env
    """

    provider = provider or settings.EMB_PROVIDER

    logger.info(f"🔸 Embeddings Provider Seleccionado: {provider}")

    if provider == "sentence_transformers":
        model = _load_local_model()
        if model is None:
            raise RuntimeError("Modelo local no disponible.")
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    elif provider == "hf":
        return _hf_embed(texts)

    elif provider == "openai":
        return _openai_embed(texts)

    else:
        raise ValueError(f"Proveedor de embeddings desconocido: {provider}")

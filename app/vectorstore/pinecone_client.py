import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from app.core.logger import logger

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "crm-rag-index")
ENV_REGION = os.getenv("PINECONE_ENV", "us-east-1")

# ============================================================
# Validación inicial
# ============================================================
if not API_KEY:
    raise RuntimeError("❌ PINECONE_API_KEY no está configurado en .env")

pc = Pinecone(api_key=API_KEY)

# ============================================================
# Crear índice
# ============================================================
def create_index(index_name: str, dim: int, metric: str = "cosine"):
    """
    Crea un índice serverless en Pinecone si no existe.
    """
    try:
        existing = pc.list_indexes().names()

        if index_name not in existing:
            logger.info(f"⚙️ Creando índice '{index_name}' en región {ENV_REGION}...")

            pc.create_index(
                name=index_name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region=ENV_REGION)
            )

            time.sleep(3)  # margen de creación
            logger.info("✅ Índice creado correctamente.")
        else:
            logger.info(f"ℹ️ El índice '{index_name}' ya existe.")

    except Exception as e:
        logger.error(f"❌ Error creando índice: {e}")
        raise

# ============================================================
# Obtener índice
# ============================================================
def get_index(index_name: str):
    """
    Devuelve una instancia de índice lista para usar.
    """
    try:
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"❌ No se pudo obtener el índice '{index_name}': {e}")
        raise

# ============================================================
# Insertar vectores
# ============================================================
def upsert_vectors(index_name: str, vectors: list):
    """
    Inserta vectores en el índice.
    Formato esperado:
    [
        (id, embedding, metadata),
        ...
    ]
    """
    try:
        index = get_index(index_name)
        index.upsert(vectors=vectors)
        logger.info(f"✅ Upsert completado: {len(vectors)} vectores insertados.")

    except Exception as e:
        logger.error(f"❌ Error durante upsert en Pinecone: {e}")
        raise

# ============================================================
# Consultar vectores
# ============================================================
def query_index(index_name: str, vector: list, top_k: int = 10,
                include_metadata: bool = True, filter: dict = None):
    """
    Realiza una consulta en Pinecone usando un embedding.
    """
    try:
        index = get_index(index_name)

        params = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata
        }

        if filter:
            params["filter"] = filter

        return index.query(**params)

    except Exception as e:
        logger.error(f"❌ Error en query: {e}")
        return {"matches": []}

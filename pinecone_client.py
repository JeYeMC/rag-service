import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time

# ==========================
# CONFIGURACIÓN E INICIALIZACIÓN
# ==========================
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-index")
ENV_REGION = os.getenv("PINECONE_ENV", "us-east-1")

# Inicializa el cliente Pinecone
pc = Pinecone(api_key=API_KEY)

# ==========================
# CREACIÓN Y GESTIÓN DE ÍNDICES
# ==========================
def create_index(index_name: str, dim: int, metric: str = "cosine"):
    """
    Crea el índice si no existe. Compatible con Pinecone serverless.
    """
    try:
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            print(f"⚙️ Creando índice '{index_name}' en {ENV_REGION}...")
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region=ENV_REGION)
            )
            print(f"✅ Índice creado correctamente: {index_name}")
            time.sleep(3)
        else:
            print(f"ℹ️ El índice '{index_name}' ya existe.")
    except Exception as e:
        print(f"❌ Error al crear índice '{index_name}': {e}")

def get_index(index_name: str):
    """Devuelve el manejador de un índice existente."""
    try:
        return pc.Index(index_name)
    except Exception as e:
        raise RuntimeError(f"Error al acceder al índice '{index_name}': {e}")

# ==========================
# INSERCIÓN (UPSERT)
# ==========================
def upsert_vectors(index_name: str, vectors: list):
    """
    Inserta o actualiza vectores con metadatos.
    vectors = [(id, vector, metadata), ...]
    """
    try:
        index = get_index(index_name)
        index.upsert(vectors=vectors)
        print(f"✅ Upsert completado en {index_name}. Total: {len(vectors)} vectores.")
    except Exception as e:
        print(f"❌ Error en upsert: {e}")

# ==========================
# CONSULTA (QUERY)
# ==========================
def query_index(index_name: str, vector, top_k=10, include_metadata=True, filter=None):
    """
    Realiza una búsqueda en Pinecone.
    Permite filtrar por metadatos, ejemplo:
    filter = {"doc_type": {"$eq": "contrato"}}
    """
    try:
        index = get_index(index_name)
        query_params = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata
        }
        if filter:
            query_params["filter"] = filter

        res = index.query(**query_params)
        return res
    except Exception as e:
        print(f"❌ Error en query: {e}")
        return {"matches": []}

# app/vectorstore/pinecone_client.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "crm-rag-index")
ENV_REGION = os.getenv("PINECONE_ENV", "us-east-1")

pc = Pinecone(api_key=API_KEY)


# -------------------------
# Crear índice
# -------------------------

def create_index(index_name: str, dim: int, metric: str = "cosine"):
    """
    Crea un índice serverless si no existe.
    """
    existing = pc.list_indexes().names()

    if index_name not in existing:
        print(f"⚙️ Creando índice '{index_name}' en región {ENV_REGION}...")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region=ENV_REGION)
        )
        time.sleep(3)
        print("✅ Índice creado.")
    else:
        print(f"ℹ️ El índice '{index_name}' ya existe.")


# -------------------------
# Obtener índice
# -------------------------

def get_index(index_name: str):
    try:
        return pc.Index(index_name)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el índice '{index_name}': {e}")


# -------------------------
# Upsert
# -------------------------

def upsert_vectors(index_name: str, vectors: list):
    """
    Inserta:
    [
        (id, embedding, metadata),
        (id, embedding, metadata),
        ...
    ]
    """
    try:
        idx = get_index(index_name)
        idx.upsert(vectors=vectors)
        print(f"✅ Upsert completado con {len(vectors)} vectores.")
    except Exception as e:
        print(f"❌ Error en upsert: {e}")


# -------------------------
# Query
# -------------------------

def query_index(index_name: str, vector, top_k=10, include_metadata=True, filter=None):
    try:
        idx = get_index(index_name)

        params = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }

        if filter:
            params["filter"] = filter

        return idx.query(**params)

    except Exception as e:
        print(f"❌ Error en query: {e}")
        return {"matches": []}

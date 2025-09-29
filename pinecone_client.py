import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-index")

# Inicializar cliente Pinecone
pc = Pinecone(api_key=API_KEY)

def create_index(index_name: str, dim: int, metric: str = "cosine"):
    # Nota: con SDK nuevo usamos .list_indexes().names
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # o "gcp"
        )
        print(f"✅ Índice creado: {index_name}")
    else:
        print(f"ℹ️ Índice {index_name} ya existe")

def get_index(index_name: str):
    return pc.Index(index_name)

def upsert_vectors(index_name: str, vectors: list):
    """vectors = [(id, vector, metadata), ...]"""
    index = get_index(index_name)
    index.upsert(vectors=vectors)
    print(f"✅ Upsert completado en {index_name}, total={len(vectors)}")

def query_index(index_name: str, vector, top_k=5, include_metadata=True):
    index = get_index(index_name)
    res = index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)
    return res

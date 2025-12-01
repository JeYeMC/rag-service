# app/vectorstore/helpers.py

import uuid

def build_metadata(
    doc_id: str,
    text: str,
    chunk_index: int,
    doc_type: str,
    source_name: str,
    extra_meta=None
):
    """
    Construye metadatos consistentes para Pinecone.
    Compatible con el RAG Pipeline completo.
    """

    base = {
        "document_id": doc_id,        # usado en queries y UI
        "chunk_index": chunk_index,   # usado para trazabilidad
        "doc_type": doc_type,         # email, contract, invoice...
        "source_name": source_name,   # upload, crm, api...
        "text_excerpt": text[:512],   # usado por reranker
        "length": len(text),
        "preview": text[:180],        # útil para debugging en dashboard
    }

    if extra_meta:
        base.update(extra_meta)

    return base


def generate_chunk_id(doc_id: str, chunk_index: int):
    """
    ID único para cada chunk.
    """
    return f"{doc_id}_chunk_{chunk_index}"


def generate_doc_id():
    """
    ID único por documento.
    """
    return str(uuid.uuid4())

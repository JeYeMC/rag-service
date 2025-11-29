# app/vectorstore/helpers.py

import uuid

def build_metadata(doc_id: str, text: str, chunk_index: int, doc_type: str, source_name: str, extra_meta=None):
    """
    Construye los metadatos consistentes para cada vector.
    """
    base = {
        "doc_id": doc_id,
        "chunk": chunk_index,
        "doc_type": doc_type,
        "source": source_name,
        "length": len(text),
        "preview": text[:180],   # ayuda para debugging
    }

    if extra_meta:
        base.update(extra_meta)

    return base


def generate_chunk_id(doc_id: str, chunk_index: int):
    """
    ID único por chunk siguiendo un patrón consistente.
    """
    return f"{doc_id}_chunk_{chunk_index}"


def generate_doc_id():
    """
    Crea un ID para cada documento.
    """
    return str(uuid.uuid4())

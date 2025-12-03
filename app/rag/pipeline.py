# app/rag/pipeline.py

import time
from typing import List, Optional
from app.core.logger import logger
from app.rag.retriever import retrieve, rerank
from app.rag.llm_router import generate_answer


# ======================================================
# 1. Templates de prompts por tipo de documento
# ======================================================
PROMPT_TEMPLATES = {
    "contrato": (
        "Eres un asistente jurídico para un CRM. Usa SOLO la información del 'Contexto' para responder. "
        "Responde en español y organiza la respuesta en: Cláusulas, Obligaciones, Vigencia, Sanciones. "
        "No cites chunk_index. Al final incluye una sección llamada 'Fuentes' con los nombres de los documentos usados."
    ),
    "correo": (
        "Eres un asistente comercial. Resume el correo en 2-4 líneas, identifica intención y acciones sugeridas. "
        "Al final incluye una sección llamada 'Fuentes' con los nombres de los documentos usados."
    ),
    "factura": (
        "Eres un asistente financiero. Extrae valores clave: total, impuestos, fechas. "
        "Al final incluye una sección llamada 'Fuentes' con los nombres de los documentos usados."
    ),
    "documento": (
        "Eres un asistente de CRM. Resume la información relevante y sugiere 1-3 acciones comerciales. "
        "Al final incluye una sección llamada 'Fuentes' con los nombres de los documentos usados."
    )
}


# ======================================================
# 2. Construcción del prompt final
# ======================================================
def build_prompt(question: str, compressed_context: List[dict], doc_type: str, documents_used: List[str]):
    """
    Construye el prompt final SIN chunk_index y con lista de documentos usados.
    """
    context_parts = [c["text"] for c in compressed_context]
    prompt_context = "\n\n".join(context_parts)

    base = PROMPT_TEMPLATES.get(doc_type, PROMPT_TEMPLATES["documento"])

    # lista de documentos
    doc_list = "\n".join(f"- {d}" for d in documents_used)

    prompt = (
        f"{base}\n\n"
        f"=== CONTEXTO ===\n{prompt_context}\n\n"
        f"=== PREGUNTA ===\n{question}\n\n"
        f"=== DOCUMENTOS DISPONIBLES COMO FUENTE ===\n{doc_list}\n\n"
        f"Responde de forma clara, profesional y usando exclusivamente la información del contexto.\n"
        f"No inventes datos. Si algo no aparece en el contexto, dilo."
    )

    return prompt


# ======================================================
# 3. Compresión del contexto (no modificar)
# ======================================================
def compress_context(hits: List[dict], max_chunks: int = 5, group_size: int = 5):
    if not hits:
        return []

    raw_texts = [h["metadata"].get("text_excerpt", "") for h in hits]
    source_infos = [
        {
            "id": h["id"],
            "chunk_index": h["metadata"].get("chunk_index"),
            "doc_type": h["metadata"].get("doc_type"),
            "document_id": h["metadata"].get("document_id"),
            "filename": h["metadata"].get("filename")
        }
        for h in hits
    ]

    groups = [raw_texts[i:i + group_size] for i in range(0, len(raw_texts), group_size)]
    src_groups = [source_infos[i:i + group_size] for i in range(0, len(source_infos), group_size)]

    compressed = []
    for grp_texts, grp_src in zip(groups, src_groups):
        joined = "\n\n".join(grp_texts)
        summary = joined[:800]

        last_dot = summary.rfind(".")
        if last_dot > 100:
            summary = summary[:last_dot + 1]

        compressed.append({"text": summary.strip(), "source_info": grp_src})

        if len(compressed) >= max_chunks:
            break

    return compressed


# ======================================================
# 4. Llamada al modelo LLM
# ======================================================
def generate_answer_with_llm(prompt: str, provider: str):
    try:
        return generate_answer(prompt, provider=provider)
    except Exception as e:
        logger.error(f"Error en generate_answer_with_llm: {e}")
        return f"⚠️ Error al generar respuesta con el modelo: {e}"


# ======================================================
# 5. Lógica principal del RAG
# ======================================================
def answer_question(
    question: str,
    top_k: int = 20,
    doc_type: Optional[str] = None,
    provider: str = "openai"
):

    start = time.time()

    # -------------------------------------------
    # Auto-detección simple
    # -------------------------------------------
    if not doc_type:
        qlow = question.lower()
        if any(w in qlow for w in ["cláusula", "contrato"]):
            doc_type = "contrato"
        elif "factura" in qlow:
            doc_type = "factura"
        elif "correo" in qlow or "email" in qlow:
            doc_type = "correo"
        else:
            doc_type = None

    # -------------------------------------------
    # Retrieve + fallback si el tipo falla
    # -------------------------------------------
    hits = retrieve(question, top_k=top_k, doc_type=doc_type, provider=provider)

    if not hits and doc_type:
        hits = retrieve(question, top_k=top_k, doc_type=None, provider=provider)
        doc_type = "documento"

    # -------------------------------------------
    # Rerank
    # -------------------------------------------
    reranked = rerank(question, hits, top_k=min(len(hits), 30), provider=provider)

    # -------------------------------------------
    # Compresión del contexto
    # -------------------------------------------
    compressed = compress_context(reranked, max_chunks=5, group_size=5)

    # -------------------------------------------
    # Filenames únicos usados
    # -------------------------------------------
    documents_used = list({h["metadata"].get("filename", "desconocido") for h in reranked})

    # -------------------------------------------
    # Construcción del prompt final
    # -------------------------------------------
    prompt = build_prompt(
        question,
        compressed,
        doc_type or "documento",
        documents_used
    )

    # -------------------------------------------
    # LLM
    # -------------------------------------------
    answer = generate_answer_with_llm(prompt, provider=provider)

    elapsed = round(time.time() - start, 2)

    return {
        "answer": answer,
        "sources": [h["metadata"] for h in hits],
        "documents_used": documents_used,
        "compressed_context": compressed,
        "doc_type": doc_type or "documento",
        "elapsed_seconds": elapsed
    }

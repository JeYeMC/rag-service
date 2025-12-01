# app/rag/pipeline.py

import time
from typing import List, Optional
from app.core.logger import logger
from app.rag.retriever import retrieve, rerank
from app.rag.ingestion import detect_document_type
from app.rag.llm_router import generate_answer   # Import correcto

PROMPT_TEMPLATES = {
    "contrato": (
        "Eres un asistente jurídico para un CRM. Usa SOLO la información del 'Contexto' para responder. "
        "Responde en español. Organiza respuesta en: Cláusulas, Obligaciones, Vigencia, Sanciones. "
        "Cita siempre fragmentos entre corchetes con el chunk_index."
    ),
    "correo": (
        "Eres un asistente comercial. Resume el correo en 2-4 líneas, identifica intención y acciones sugeridas."
    ),
    "factura": (
        "Eres un asistente financiero. Extrae valores clave: total, impuestos, fechas."
    ),
    "documento": (
        "Eres un asistente de CRM. Resume la información relevante y sugiere 1-3 acciones comerciales."
    )
}

def build_prompt(question: str, compressed_context: List[dict], doc_type: str):
    context_parts = []
    for c in compressed_context:
        idxs = [s.get("chunk_index") for s in c.get("source_info", []) if s.get("chunk_index") is not None]
        idxs = ",".join(str(int(i)) for i in idxs) if idxs else ""
        marker = f"[chunks: {idxs}]\n" if idxs else ""
        context_parts.append(f"{marker}{c['text']}")

    prompt_context = "\n\n".join(context_parts)
    base = PROMPT_TEMPLATES.get(doc_type, PROMPT_TEMPLATES["documento"])

    prompt = (
        f"{base}\n\n"
        f"Contexto:\n{prompt_context}\n\n"
        f"Pregunta: {question}\n\n"
        f"Respuesta estructurada (incluye fuentes):"
    )

    return prompt


def compress_context(hits: List[dict], max_chunks: int = 5, group_size: int = 5):
    if not hits:
        return []

    raw_texts = [h["metadata"].get("text_excerpt", "") for h in hits]
    source_infos = [
        {
            "id": h["id"],
            "chunk_index": h["metadata"].get("chunk_index"),
            "doc_type": h["metadata"].get("doc_type"),
            "document_id": h["metadata"].get("document_id")
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


def generate_answer_with_llm(prompt: str, provider: str):
    try:
        return generate_answer(prompt, provider=provider)
    except Exception as e:
        logger.error(f"Error en generate_answer_with_llm: {e}")
        return f"⚠️ Error al generar respuesta con el modelo: {e}"


def answer_question(
    question: str,
    top_k: int = 20,
    doc_type: Optional[str] = None,
    provider: str = "openai"
):

    start = time.time()

    # Auto-detección del tipo
    if not doc_type:
        qlow = question.lower()
        if any(w in qlow for w in ["cláusula", "contrato"]):
            doc_type = "contrato"
        elif "factura" in qlow:
            doc_type = "factura"
        else:
            doc_type = None

    hits = retrieve(question, top_k=top_k, doc_type=doc_type)

    if not hits and doc_type:
        hits = retrieve(question, top_k=top_k, doc_type=None)
        doc_type = "documento"

    reranked = rerank(question, hits, top_k=min(len(hits), 30))

    compressed = compress_context(reranked, max_chunks=5, group_size=5)

    prompt = build_prompt(question, compressed, doc_type or "documento")

    answer = generate_answer_with_llm(prompt, provider=provider)

    elapsed = round(time.time() - start, 2)

    return {
        "answer": answer,
        "sources": [h["metadata"] for h in hits],
        "compressed_context": compressed,
        "doc_type": doc_type or "documento",
        "elapsed_seconds": elapsed
    }

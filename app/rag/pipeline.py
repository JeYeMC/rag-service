# app/rag/pipeline.py
import time
from typing import List, Optional
from app.core.logger import logger
from app.core.config import settings
from app.rag.retriever import retrieve, rerank
from app.rag.ingestion import detect_document_type

# Prompt templates (puedes mover a app/rag/prompts.py si prefieres)
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
    prompt = f"{base}\n\nContexto:\n{prompt_context}\n\nPregunta: {question}\n\nRespuesta estructurada (incluye fuentes):"
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
        } for h in hits
    ]

    groups = [raw_texts[i:i + group_size] for i in range(0, len(raw_texts), group_size)]
    src_groups = [source_infos[i:i + group_size] for i in range(0, len(source_infos), group_size)]

    compressed = []
    for grp_texts, grp_src in zip(groups, src_groups):
        joined = "\n\n".join(grp_texts)
        # simple compression (puedes usar summarizer si lo configuras)
        summary = joined[:800]
        last_dot = summary.rfind(".")
        if last_dot > 100:
            summary = summary[:last_dot + 1]
        compressed.append({"text": summary.strip(), "source_info": grp_src})
        if len(compressed) >= max_chunks:
            break
    return compressed

def generate_answer_with_llm(prompt: str) -> str:
    """
    Llama al LLM configurado (HF Router, HF Inference o fallback).
    Aquí mantengo una llamada simple a HF Router si está configurada.
    """
    import requests
    from app.core.config import settings

    if settings.HF_API_URL and "router.huggingface.co" in settings.HF_API_URL:
        headers = {"Authorization": f"Bearer {settings.HF_INFERENCE_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": settings.HF_MODEL or "",
            "messages": [
                {"role": "system", "content": "Asistente CRM."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.2
        }
        r = requests.post(settings.HF_API_URL, headers=headers, json=payload, timeout=180)
        if r.status_code == 200:
            try:
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                return str(r.json())
        return f"LLM error: {r.status_code} {r.text}"

    # Fallback: no LLM configurado, devolver el prompt (útil para pruebas)
    return "⚠️ No hay modelo LLM configurado. Prompt generado:\n\n" + prompt

def answer_question(question: str, top_k: int = 20, doc_type: Optional[str] = None):
    start = time.time()
    # si no viene doc_type, detectamos por pregunta como heurística
    if not doc_type:
        qlow = question.lower()
        if any(w in qlow for w in ["cláusula", "contrato"]):
            doc_type = "contrato"
        elif "factura" in qlow:
            doc_type = "factura"
        else:
            doc_type = None

    hits = retrieve(question, top_k=top_k, doc_type=doc_type)
    # si no hay resultados y había filtro, intentar global
    if not hits and doc_type:
        hits = retrieve(question, top_k=top_k, doc_type=None)
        doc_type = "documento"

    reranked = rerank(question, hits, top_k=min(len(hits), 30))
    compressed = compress_context(reranked, max_chunks=5, group_size=5)

    prompt = build_prompt(question, compressed, doc_type or "documento")
    answer = generate_answer_with_llm(prompt)

    elapsed = round(time.time() - start, 2)
    return {
        "answer": answer,
        "sources": [h["metadata"] for h in hits],
        "compressed_context": compressed,
        "doc_type": doc_type or "documento",
        "elapsed_seconds": elapsed
    }

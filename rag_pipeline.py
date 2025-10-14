import os
import uuid
import requests
import openai
import time
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

from ingest_utils import extract_text, chunk_text
from pinecone_client import create_index, upsert_vectors, query_index

load_dotenv()

# ================== CONFIGURACIÓN ==================
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-index")
EMB_MODEL = os.getenv("EMB_MODEL", "all-mpnet-base-v2")

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf_inference")
HF_API_URL = os.getenv("HF_API_URL", "")
HF_MODEL = os.getenv("HF_MODEL", "")
HF_KEY = os.getenv("HF_INFERENCE_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

# ================== EMBEDDINGS ==================
emb_model = SentenceTransformer(EMB_MODEL)
VECTOR_DIM = emb_model.get_sentence_embedding_dimension()

def embed_texts(texts):
    vectors = emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()

# ================== RE-RANKER Y SUMMARIZER ==================
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

USE_LOCAL_SUMMARIZER = os.getenv("USE_LOCAL_SUMMARIZER", "false").lower() == "true"
if USE_LOCAL_SUMMARIZER:
    summarizer = pipeline("summarization", model=os.getenv("SUMMARIZER_MODEL", "google/pegasus-xsum"))
else:
    summarizer = None

# ================== DETECCIÓN DE TIPO DE DOCUMENTO ==================
def detect_doc_type(text: str) -> str:
    """
    Clasifica automáticamente el tipo de documento con reglas simples.
    Puedes ampliar con un clasificador ML si quieres más precisión.
    """
    t = text.lower()
    # patterns for contracts
    if any(w in t for w in ["cláusula", "clausula", "contrato", "objeto del contrato", "cláusulas"]):
        return "contrato"
    # patterns for email-like
    if any(w in t for w in ["estimado", "cordial saludo", "atentamente", "saludos", "buenos días", "buenas tardes", "enviado"]):
        return "correo_cliente"
    # invoice
    if any(w in t for w in ["factura", "subtotal", "iva", "valor total", "número de factura"]):
        return "factura"
    # PQRS / solicitud / reclamo
    if any(w in t for w in ["pqrs", "petición", "reclamo", "queja", "solicitud"]):
        return "pqrs"
    # proposal
    if any(w in t for w in ["propuesta", "servicios ofrecidos", "alcance", "entregables"]):
        return "propuesta"
    return "otro"

# ================== INGESTA (ahora con doc_type) ==================
def ingest_file_to_pinecone(file_path: str, source_name: str = "upload", chunk_size: int = 500):
    """
    Extrae texto, detecta tipo, lo parte en chunks, genera embeddings y los sube a Pinecone.
    Añade metadata 'doc_type' para búsquedas filtradas.
    """
    text = extract_text(file_path)
    if not text or not text.strip():
        return {"error": "No se pudo extraer texto"}

    doc_type = detect_doc_type(text)

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=100)
    vectors = embed_texts(chunks)

    upserts = []
    for i, vec in enumerate(vectors):
        doc_id = str(uuid.uuid4())
        metadata = {
            "source": source_name,
            "chunk_index": i,
            "text_excerpt": chunks[i][:500],
            "doc_type": doc_type
        }
        upserts.append((doc_id, vec, metadata))

    create_index(INDEX_NAME, dim=VECTOR_DIM)
    upsert_vectors(INDEX_NAME, upserts)

    return {"status": "ok", "ingested_chunks": len(upserts), "source": source_name, "doc_type": doc_type}


# ================== RECUPERACIÓN (filtrado por tipo opcional) ==================
def retrieve(query: str, top_k=20, doc_type: str = None):
    """
    Busca los chunks más relevantes desde Pinecone.
    Si doc_type está presente, intenta filtrar los resultados por metadata doc_type.
    Para compatibilidad con un cliente Pinecone simple, hacemos post-filtering.
    """
    qvec = embed_texts([query])[0]
    # pedir más (pool) para luego filtrar mejor
    pool_k = max(top_k * 4, 50)
    res = query_index(INDEX_NAME, qvec, top_k=pool_k)
    hits = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {}) or {}
        if doc_type:
            # si el metadata tiene doc_type y coincide, lo aceptamos;
            # si no tiene doc_type lo rechazamos (opcional: podrías aceptar si no viene)
            if meta.get("doc_type") and meta.get("doc_type") != doc_type:
                continue
        hits.append({
            "id": m.get("id"),
            "score": m.get("score"),
            "metadata": meta
        })
    # ordenar por score original (o por metadata if present)
    hits_sorted = sorted(hits, key=lambda x: x["score"], reverse=True)
    return hits_sorted[:top_k]


# ================== RE-RANKING ==================
def rerank(query: str, hits: list, top_k: int = 10):
    """Reordena los resultados con CrossEncoder para mejorar precisión."""
    if not hits:
        return []
    pairs = [(query, h["metadata"].get("text_excerpt", "")) for h in hits]
    try:
        scores = cross_encoder.predict(pairs)
    except Exception as e:
        # si falla el cross-encoder, devolvemos hits sin cambios
        print("Cross-encoder error:", e)
        return hits[:top_k]
    for i, h in enumerate(hits):
        h["_rerank_score"] = float(scores[i]) if i < len(scores) else 0.0
    hits_sorted = sorted(hits, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
    return hits_sorted[:top_k]


# ================== COMPRESIÓN DE CONTEXTO ==================
def compress_context(hits: list, max_chunks: int = 5, group_size: int = 5):
    """
    Agrupa y resume chunks para reducir el contexto antes del LLM.
    Devuelve una lista de {'text': summary, 'source_info': [...]}
    """
    if not hits:
        return []

    raw_texts = [h["metadata"].get("text_excerpt", "") for h in hits]
    source_infos = [
        {
            "id": h["id"],
            "chunk_index": h["metadata"].get("chunk_index"),
            "source": h["metadata"].get("source"),
            "doc_type": h["metadata"].get("doc_type")
        } for h in hits
    ]

    groups = [raw_texts[i:i + group_size] for i in range(0, len(raw_texts), group_size)]
    src_groups = [source_infos[i:i + group_size] for i in range(0, len(source_infos), group_size)]

    compressed = []
    for grp_texts, grp_src in zip(groups, src_groups):
        joined = "\n\n".join(grp_texts)
        if USE_LOCAL_SUMMARIZER and summarizer:
            try:
                out = summarizer(joined, max_length=128, min_length=30, do_sample=False)
                summary = out[0]["summary_text"] if isinstance(out, list) else str(out)
            except Exception as e:
                print("Summarizer error:", e)
                summary = joined[:800]
        else:
            # fallback: simple compression by truncation keeping sentence boundaries
            # trata de cortar por punto para no dejar frases incompletas
            summary = joined[:800]
            last_dot = summary.rfind(".")
            if last_dot > int(len(summary) * 0.6):
                summary = summary[:last_dot+1]
        compressed.append({"text": summary.strip(), "source_info": grp_src})
        if len(compressed) >= max_chunks:
            break
    return compressed


# ================== PROMPTS ADAPTABLES POR TIPO ==================
PROMPT_TEMPLATES = {
    "contrato": (
        "Eres un asistente jurídico para un CRM. Usa SOLO la información del 'Contexto' para responder. "
        "Responde en español de forma formal, organiza la respuesta en secciones (ej. Cláusulas, Obligaciones, Duración, Sanciones). "
        "Incluye referencias a los fragmentos del documento entre corchetes indicando el número de chunk cuando cites contenido."
    ),
    "correo_cliente": (
        "Eres un asistente comercial para un CRM. Resume el correo del cliente en 2-4 líneas, indica el tono (ej. cordial, urgente), "
        "la intención principal y las acciones recomendadas."
    ),
    "factura": (
        "Eres un asistente financiero. Extrae valores clave (valor total, impuestos, fechas de emisión y vencimiento) y presenta en formato claro."
    ),
    "pqrs": (
        "Eres un agente de atención al cliente. Clasifica la solicitud (Petición/Queja/Reclamo/Sugerencia), resume el motivo y sugiere la acción de respuesta."
    ),
    "propuesta": (
        "Eres un asistente comercial. Resume la propuesta en secciones: servicios ofrecidos, entregables, duración y condiciones económicas."
    ),
    "otro": (
        "Eres un asistente de CRM. Resume la información principal del documento y sugiere acciones prácticas para el equipo comercial."
    )
}

CRM_SYSTEM_INSTRUCTIONS = (
    "Eres un asistente para un CRM empresarial. Usa SOLO la información de 'Contexto' para responder. "
    "Responde en español. Para cada afirmación derivada, cita la fuente en formato [chunk_index]. "
    "Finalmente, sugiere 1–3 acciones comerciales (seguimiento, propuesta, escalado legal, etc.) "
    "y extrae entidades clave (Cliente, Fechas, Valores, Penalizaciones) si existen.\n\n"
)

def build_prompt(question: str, compressed_context: list, doc_type: str = "otro"):
    """Construye el prompt con instrucciones CRM y contexto comprimido, adaptado al tipo de documento."""
    base = PROMPT_TEMPLATES.get(doc_type, PROMPT_TEMPLATES["otro"])
    context_parts = []
    for c in compressed_context:
        idxs = [s.get("chunk_index") for s in c.get("source_info", []) if s.get("chunk_index") is not None]
        # formatear índices como enteros y sin decimales
        idxs_formatted = ",".join(str(int(i)) for i in idxs) if idxs else ""
        marker = f"[chunks: {idxs_formatted}]\n" if idxs_formatted else ""
        context_parts.append(f"{marker}{c['text']}")
    prompt_context = "\n\n".join(context_parts)
    prompt = (
        f"{base}\n\nContexto:\n{prompt_context}\n\nPregunta: {question}\n\n"
        "Respuesta estructurada (incluye fuentes entre corchetes si aplica):"
    )
    return prompt

# ================== GENERACIÓN DE RESPUESTA (CRM-aware pipeline) ==================
def generate_answer(question: str, context_hits: list, doc_type: str = "otro", k_rerank: int = 20, k_final: int = 5):
    """
    Estrategia:
      - Re-rank top k_rerank
      - Comprimir contexto a k_final unidades
      - Construir prompt adaptado al doc_type
      - Llamar LLM (HuggingFace Router, HF Inference o OpenAI)
    """
    start = time.time()

    reranked = rerank(question, context_hits, top_k=min(k_rerank, len(context_hits)))
    compressed = compress_context(reranked, max_chunks=k_final, group_size=5)
    prompt = build_prompt(question, compressed, doc_type=doc_type)

    # Llamada al LLM
    output = ""
    if HF_API_URL and "router.huggingface.co" in HF_API_URL:
        headers = {"Authorization": f"Bearer {HF_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": CRM_SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.2
        }
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            try:
                output = data["choices"][0]["message"]["content"]
            except Exception:
                output = str(data)
        else:
            output = f"LLM error: {resp.status_code} {resp.text}"
    elif HF_API_URL and "api-inference.huggingface.co" in HF_API_URL:
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        data = {"inputs": prompt, "parameters": {"max_new_tokens": 256}}
        resp = requests.post(HF_API_URL, headers=headers, json=data, timeout=120)
        if resp.status_code == 200:
            o = resp.json()
            output = o[0].get("generated_text", "") if isinstance(o, list) else str(o)
        else:
            output = f"LLM error: {resp.status_code} {resp.text}"
    elif LLM_PROVIDER == "openai" and OPENAI_KEY:
        openai.api_key = OPENAI_KEY
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": CRM_SYSTEM_INSTRUCTIONS}, {"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2
        )
        output = completion.choices[0].message.content
    else:
        output = "⚠️ No hay modelo configurado o la URL es inválida."

    elapsed = round(time.time() - start, 2)
    diagnostics = {
        "reranked_count": len(reranked),
        "compressed_units": len(compressed),
        "elapsed_seconds": elapsed
    }
    return output, diagnostics, compressed

# ================== LIMPIEZA Y POST-PROCESADO ==================
def clean_answer(answer: str) -> str:
    """Limpia y normaliza la respuesta del LLM para quitar ruidos y mejorar legibilidad."""
    if not isinstance(answer, str):
        answer = str(answer)
    # Normaliza floats de índices tipo '61.0' -> '61'
    answer = re.sub(r'(\d+)\.0\b', r'\1', answer)
    # Cambia a guiones para listas
    answer = answer.replace('*', '-').replace('•', '-')
    # Asegura saltos de línea dobles para separación clara
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    # Quita espacios múltiples
    answer = re.sub(r'[ \t]+', ' ', answer)
    # Trim
    return answer.strip()

# ================== DETECCIÓN TIPO DE PREGUNTA (para filtrar) ==================
def detect_query_type(query: str) -> str:
    q = query.lower()
    if "cláusula" in q or "contrato" in q or "clausula" in q:
        return "contrato"
    if "correo" in q or "cliente" in q or "mensaje" in q:
        return "correo_cliente"
    if "factura" in q or "pago" in q or "vencim" in q:
        return "factura"
    if "reclamo" in q or "pqrs" in q or "queja" in q:
        return "pqrs"
    if "propuesta" in q or "servicios" in q:
        return "propuesta"
    return "otro"

# ================== PIPELINE COMPLETO (expuesta al app.py) ==================
def answer_question(question: str, top_k: int = 20):
    """
    - Detecta tipo de pregunta (y por tanto doc_type)
    - Recupera (filtra por tipo) y re-rankea
    - Genera respuesta estructurada y limpia
    """
    query_type = detect_query_type(question)
    # Si detectamos doc_type 'otro', recuperamos sin filtrar, else intentamos filtrar
    doc_type_filter = query_type if query_type != "otro" else None

    hits = retrieve(question, top_k=top_k, doc_type=doc_type_filter)
    answer_raw, diagnostics, compressed = generate_answer(question, hits, doc_type=doc_type_filter or "otro",
                                                          k_rerank=top_k, k_final=5)
    answer = clean_answer(answer_raw)

    # prepare sources: include top-k original hits metadata (already filtered)
    sources = [h["metadata"] for h in hits[:top_k]]

    return {
        "answer": answer,
        "sources": sources,
        "diagnostics": diagnostics,
        "compressed_context": compressed,
        "doc_type": doc_type_filter or "otro"
    }

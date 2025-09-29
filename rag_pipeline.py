import os
import uuid
import requests
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ingest_utils import extract_text, chunk_text
from pinecone_client import create_index, upsert_vectors, query_index

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-index")
EMB_PROVIDER = os.getenv("EMB_PROVIDER", "sentence_transformers")
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf_inference")

# Embeddings
if EMB_PROVIDER == "sentence_transformers":
    emb_model = SentenceTransformer(EMB_MODEL)
    VECTOR_DIM = emb_model.get_sentence_embedding_dimension()
else:
    raise ValueError("Proveedor de embeddings no soportado todavía")

def embed_texts(texts):
    vectors = emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()

def ingest_file_to_pinecone(file_path: str, source_name: str = "upload", chunk_size: int = 500):
    text = extract_text(file_path)
    if not text.strip():
        return {"error": "No se pudo extraer texto"}

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=100)
    vectors = embed_texts(chunks)

    upserts = []
    for i, vec in enumerate(vectors):
        doc_id = str(uuid.uuid4())
        metadata = {
            "source": source_name,
            "chunk_index": i,
            "text_excerpt": chunks[i][:200]
        }
        upserts.append((doc_id, vec, metadata))

    create_index(INDEX_NAME, dim=VECTOR_DIM)
    upsert_vectors(INDEX_NAME, upserts)

    return {"status": "ok", "ingested_chunks": len(upserts), "source": source_name}

def retrieve(query: str, top_k=3):
    qvec = embed_texts([query])[0]
    res = query_index(INDEX_NAME, qvec, top_k=top_k)
    hits = []
    for m in res["matches"]:
        hits.append({
            "id": m["id"],
            "score": m["score"],
            "metadata": m.get("metadata", {})
        })
    return hits

def generate_answer(question: str, context_chunks: list):
    prompt_context = "\n".join(
        [f"- {c['metadata'].get('text_excerpt','')}" for c in context_chunks]
    )
    prompt = (
        f"Usa el siguiente contexto para responder en español.\n\n"
        f"Contexto:\n{prompt_context}\n\n"
        f"Pregunta: {question}\n\n"
        f"Respuesta:"
    )

    if LLM_PROVIDER == "hf_inference":
        hf_key = os.getenv("HF_INFERENCE_API_KEY")
        api_url = os.getenv("HF_API_URL")
        headers = {"Authorization": f"Bearer {hf_key}"}
        data = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 256, "temperature": 0.3}
        }
        resp = requests.post(api_url, headers=headers, json=data, timeout=60)

        if resp.status_code == 200:
            output = resp.json()
            # flan-t5 siempre devuelve una lista con "generated_text"
            if isinstance(output, list) and len(output) > 0:
                return output[0].get("generated_text", "").strip()
            elif isinstance(output, dict) and "generated_text" in output:
                return output["generated_text"].strip()
            else:
                return "⚠️ El modelo no devolvió texto."
        else:
            return f"LLM error: {resp.status_code} {resp.text}"

    elif LLM_PROVIDER == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )
        return completion.choices[0].message.content

    else:
        return "⚠️ No hay LLM configurado."

def answer_question(question: str, top_k=3):
    hits = retrieve(question, top_k=top_k)
    answer = generate_answer(question, hits)
    return {
        "answer": answer,
        "sources": [h["metadata"] for h in hits]
    }

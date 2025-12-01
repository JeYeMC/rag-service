# app/rag/llm_router.py

import requests
from app.core.logger import logger
from app.core.config import settings

# ======================================================
# 🔥 GENERADOR DE RESPUESTAS (Router HF / OpenAI)
# ======================================================

def _call_hf_chat(prompt: str) -> str:
    """
    Llama a HuggingFace Inference API (chat/completions).
    """
    if not settings.HF_INFERENCE_API_KEY or not settings.HF_MODEL:
        raise RuntimeError("Variables HF no configuradas.")

    url = settings.HF_API_URL or f"https://api-inference.huggingface.co/models/{settings.HF_MODEL}"

    headers = {
        "Authorization": f"Bearer {settings.HF_INFERENCE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.2
        }
    }

    logger.info(f"🧠 Llamando HF Chat: {settings.HF_MODEL}")

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HF Error: {r.text}")

    try:
        return r.json()[0]["generated_text"]
    except:
        return str(r.json())


def _call_openai_chat(prompt: str) -> str:
    """
    Llama al modelo OpenAI (gpt-4o-mini o el que configures).
    Compatible con el nuevo SDK (2024+).
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no definido.")

    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    logger.info(f"🧠 Llamando OpenAI Chat ({settings.OPENAI_MODEL})")

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Asistente experto en análisis de documentos para CRM."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.2
    )

    # Nuevo acceso correcto
    return response.choices[0].message.content


# ======================================================
# 🔥 RESUMENES
# ======================================================

def generate_summary(text: str, provider: str = None) -> str:
    """
    Resume un documento usando el proveedor seleccionado.
    """
    provider = provider or settings.LLM_PROVIDER
    logger.info(f"📘 Generando resumen con provider='{provider}'")

    prompt = (
        "Resume el siguiente documento de forma clara, en máximo 10 líneas. "
        "Devuelve un resumen organizado y conciso.\n\n"
        f"Documento:\n{text[:6000]}"
    )

    try:
        if provider == "openai":
            return _call_openai_chat(prompt)
        else:
            return _call_hf_chat(prompt)
    except Exception as e:
        logger.error(f"Resumen falló: {e}")
        return text[:1200]  # fallback


# ======================================================
# 🔥 RESPUESTA LARGA PARA RAG
# ======================================================

def generate_answer(prompt: str, provider: str = None) -> str:
    """
    Genera una respuesta del LLM usando OpenAI o HuggingFace.
    """
    provider = provider or settings.LLM_PROVIDER
    logger.info(f"🤖 Generando respuesta LLM con provider='{provider}'")

    try:
        if provider == "openai":
            return _call_openai_chat(prompt)
        else:
            return _call_hf_chat(prompt)
    except Exception as e:
        logger.error(f"Error LLM: {e}")
        return "⚠️ Error al llamar al modelo LLM.\n" + prompt

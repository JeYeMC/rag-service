# app/rag/prompts.py

"""
Prompts usados por el RAG Pipeline.
Incluye contexto CRM-aware y especialización según tipo de documento.
"""

# --------------------------
# 1. Prompt general del sistema
# --------------------------

BASE_SYSTEM_PROMPT = """
Eres un asistente experto en análisis de documentos para un CRM empresarial.
Debes responder de forma clara, objetiva y con trazabilidad hacia el documento original.

Sigue estas reglas estrictamente:
- Usa únicamente la información del contexto recuperado.
- Si falta información, dilo explícitamente.
- No inventes datos inexistentes.
- Si hay ambigüedad, pide aclaración.
- Ajusta tu estilo dependiendo del tipo de documento (email, contrato, factura, etc.).
"""

# --------------------------
# 2. Prompts especializados por tipo de documento
# --------------------------

DOC_TYPE_PROMPTS = {
    "contract": """
El documento es un **contrato**.
Extrae y explica:
- Partes contratantes
- Objeto del contrato
- Plazos / vigencias
- Responsabilidades
- Costos u obligaciones económicas
- Cláusulas relevantes (penalidades, terminación, confidencialidad)
- Riesgos o alertas si se detectan inconsistencias.
""",

    "email": """
El documento es un **correo electrónico**.
Identifica:
- Quién escribe y a quién
- Intención principal
- Solicitudes específicas
- Tono (formal, urgente, seguimiento, reclamo, etc.)
- Acciones requeridas
""",

    "invoice": """
El documento es una **factura**.
Extrae:
- Cliente
- Fecha
- Valor total
- Ítems facturados
- Impuestos
- Número de factura
""",

    "pqr": """
El documento es una **PQRS** (petición, queja, reclamo o solicitud).
Identifica:
- Tipo de solicitud
- Usuario afectado
- Problema o requerimiento
- Nivel de urgencia
- Respuesta recomendada
""",

    "policy": """
El documento es una **política o reglamento corporativo**.
Describe:
- Objetivo de la política
- Normas clave
- Roles/responsables
- Casos de uso o aplicación
""",

    "other": """
Documento general.
Extrae la información relevante y responde de forma clara usando únicamente el contexto.
"""
}

# --------------------------
# 3. Prompt para resumen avanzado
# --------------------------

SUMMARY_PROMPT = """
Resume el siguiente contenido de forma clara y estructurada.
Incluye:
- Idea principal
- Puntos claves
- Entidades importantes
- Fechas, valores o compromisos si los hay

No inventes información no presente en el contenido.
"""

# --------------------------
# 4. Prompt para extracción avanzada de entidades
# --------------------------

ENTITY_EXTRACTION_PROMPT = """
Extrae entidades estructuradas del siguiente contenido.
Devuelve el resultado en formato JSON:

{
  "personas": [],
  "organizaciones": [],
  "fechas": [],
  "valores_economicos": [],
  "productos_servicios": [],
  "lugares": [],
  "acciones_solicitadas": []
}

No inventes entidades que no estén explícitamente en el contenido.
"""

# --------------------------
# 5. Helper para obtener el prompt final
# --------------------------

def build_prompt(doc_type: str, context: str, question: str):
    """
    Construye el prompt completo para el LLM basándose en:
    - prompt base
    - prompt especializado según doc_type
    """
    doc_prompt = DOC_TYPE_PROMPTS.get(doc_type, DOC_TYPE_PROMPTS["other"])

    return f"""
{BASE_SYSTEM_PROMPT}

=== Información del documento (contexto relevante) ===
{context}

=== Instrucciones según tipo de documento ({doc_type}) ===
{doc_prompt}

=== Pregunta del usuario ===
{question}

Responde de forma clara, profesional y usando solamente la información del contexto.
"""

"""
Prompts usados por el RAG Pipeline.
Versión ajustada: sin chunk_index y con reporte de documentos usados.
"""

# ============================================================
# 1. Prompt general del sistema (más estricto, sin chunk_index)
# ============================================================

BASE_SYSTEM_PROMPT = """
Eres un asistente experto en análisis documental dentro de un CRM empresarial,
especializado en contratos, facturas, correos, políticas y documentos corporativos.

Debes responder SIEMPRE usando exclusivamente la información del CONTEXTO provisto.
No puedes usar conocimientos externos, memoria previa ni completar huecos inventados.

REGLAS OBLIGATORIAS:
- Usa ÚNICAMENTE el *Contexto* provisto.
- NO inventes datos. No asumas nada.
- Si no encuentras la información en el contexto: responde “El documento no contiene esa información según el contexto disponible.”
- Si hay ambigüedad: solicita aclaración.
- NO uses chunk_index ni IDs internos.
- Al final incluye una sección llamada **Fuentes**, listando los nombres de los documentos usados.
- Mantén las respuestas claras, breves y basadas únicamente en el contenido del contexto.
- Nunca cambies el significado del texto original.
"""

# ============================================================
# 2. Prompts especializados por tipo de documento
# ============================================================

DOC_TYPE_PROMPTS = {
    "contract": """
El documento es un **contrato**. Extrae y explica únicamente lo explícito:
- Partes contratantes
- Objeto del contrato
- Plazo o vigencia
- Obligaciones principales
- Condiciones económicas
- Cláusulas de terminación
- Penalidades o sanciones
- Responsabilidades sobre propiedad intelectual
- Alertas sobre riesgos documentados

No interpretes más allá del texto.
""",

    "email": """
El documento es un **correo electrónico**. Identifica:
- Remitente, destinatario(s)
- Motivo del mensaje
- Solicitudes
- Urgencia si se menciona
- Compromisos o acciones solicitadas
""",

    "invoice": """
El documento es una **factura**. Extrae solo lo explícito:
- Cliente
- Fecha de emisión
- Valor total
- Ítems facturados
- Impuestos
- Número de factura
""",

    "pqr": """
El documento corresponde a una **PQRS**. Extrae:
- Tipo (petición, queja, reclamo o solicitud)
- Usuario afectado
- Descripción del problema
- Urgencia si se menciona
- Posible respuesta basada únicamente en el texto
""",

    "policy": """
El documento es una **política corporativa**. Describe:
- Objetivo
- Normas principales
- Responsables
- Ámbito de aplicación
""",

    "other": """
Documento general.
Extrae únicamente la información explícita relevante.
"""
}

# ============================================================
# 3. Prompt para resumen avanzado
# ============================================================

SUMMARY_PROMPT = """
Resume el contenido de forma objetiva, concisa y sin agregar información externa.

Incluye:
- Idea principal
- Puntos clave
- Entidades relevantes
- Fechas o valores si aparecen

Regla clave: NO inventes información.
"""

# ============================================================
# 4. Prompt para extracción avanzada de entidades
# ============================================================

ENTITY_EXTRACTION_PROMPT = """
Extrae entidades SOLO si aparecen explícitamente.
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

Si una categoría no tiene información explícita, déjala como lista vacía.
"""

# ============================================================
# 5. Helper para construir prompts finales (sin chunk_index)
# ============================================================

def build_prompt(doc_type: str, context: str, question: str):
    """
    Construye el prompt:
    - Prompt base del sistema
    - Prompt especializado por tipo de documento
    - Contexto
    - Pregunta del usuario

    NOTA: Este prompt ya NO usa chunk_index.
    """

    doc_prompt = DOC_TYPE_PROMPTS.get(doc_type, DOC_TYPE_PROMPTS["other"])

    return f"""
{BASE_SYSTEM_PROMPT}

=== CONTEXTO DEL DOCUMENTO (fragmentos recuperados) ===
{context}

=== INSTRUCCIONES ESPECÍFICAS PARA TIPO DE DOCUMENTO: {doc_type.upper()} ===
{doc_prompt}

=== PREGUNTA DEL USUARIO ===
{question}

Recuerda:
- Usa únicamente el CONTEXTO.
- No incluyas chunk_index.
- No inventes información.
- Al final el LLM debe agregar una sección 'Fuentes' con los nombres de los documentos usados.
"""

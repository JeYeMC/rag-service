from pdfminer.high_level import extract_text as extract_pdf_text
import docx
import email
from pathlib import Path
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==========================
# EXTRACCIÓN DE TEXTO
# ==========================
def extract_text_from_pdf(path: str) -> str:
    """Extrae texto de un PDF, eliminando saltos y espacios innecesarios."""
    text = extract_pdf_text(path)
    return clean_text(text)

def extract_text_from_docx(path: str) -> str:
    """Extrae texto de un archivo DOCX."""
    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return clean_text(text)

def extract_text_from_txt(path: str) -> str:
    """Lee texto de un archivo plano UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return clean_text(text)

def extract_text_from_email(path: str) -> str:
    """Extrae texto del cuerpo de un correo electrónico .eml."""
    with open(path, "r", encoding="utf-8") as f:
        msg = email.message_from_file(f)
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_payload(decode=True).decode(errors="ignore"))
    else:
        try:
            parts.append(msg.get_payload(decode=True).decode(errors="ignore"))
        except Exception:
            parts.append(msg.get_payload())
    return clean_text("\n".join(parts))

def extract_text(path: str) -> str:
    """Detecta el tipo de archivo y aplica el extractor correspondiente."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".eml":
        return extract_text_from_email(path)
    else:
        raise ValueError(f"❌ Formato no soportado: {ext}")

# ==========================
# LIMPIEZA DE TEXTO
# ==========================
def clean_text(text: str) -> str:
    """Normaliza texto, eliminando saltos de línea repetidos, espacios y numeración ruidosa."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    # eliminar numeraciones tipo “1.” o “1)” al inicio de línea
    text = re.sub(r"(?m)^\s*\d+[\.\)]\s*", "", text)
    text = text.strip()
    return text

# ==========================
# DIVISIÓN INTELIGENTE DE TEXTO (CHUNKS)
# ==========================
def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200):
    """
    Divide texto en fragmentos semánticos:
    - Detecta estructuras típicas de contratos, correos, facturas o propuestas
    - Usa LangChain TextSplitter si no hay delimitadores específicos
    """

    # 1️⃣ Si es un contrato, intenta cortar por “CLÁUSULA” o “CLAUSULA”
    clausulas = re.split(r"(?i)\b(CLÁUSULA|CLAUSULA)\b", text)
    if len(clausulas) > 3:
        chunks = []
        for i in range(0, len(clausulas), 2):
            seg = clausulas[i : i + 2]
            if len(seg) == 2:
                combined = f"{seg[0].strip()} {seg[1].strip()}"
            else:
                combined = seg[0]
            if len(combined) > 50:
                chunks.append(combined.strip())
        return chunks

    # 2️⃣ Si parece un correo electrónico, corta por separadores comunes
    if re.search(r"(?i)(asunto|estimad|saludo|atentamente|firma)", text):
        parts = re.split(r"(?i)(de:|para:|asunto:|atentamente|saludos|firma)", text)
        return [p.strip() for p in parts if len(p.strip()) > 80]

    # 3️⃣ Si parece una factura, corta por líneas de totales o ítems
    if re.search(r"(?i)(subtotal|iva|total|valor|factura)", text):
        parts = re.split(r"(?i)(subtotal|iva|total|valor total)", text)
        return [p.strip() for p in parts if len(p.strip()) > 80]

    # 4️⃣ Si parece una propuesta, corta por secciones (“Alcance”, “Entregables”, “Condiciones”)
    if re.search(r"(?i)(alcance|entregables|condiciones|ofrecemos|duración|valor)", text):
        parts = re.split(r"(?i)(alcance|entregables|condiciones|duración|valor|plazo)", text)
        return [p.strip() for p in parts if len(p.strip()) > 80]

    # 5️⃣ Si nada aplica, usa el splitter genérico de LangChain
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", ";", " "]
    )
    docs = splitter.create_documents([text])
    return [d.page_content.strip() for d in docs if d.page_content.strip()]
from pdfminer.high_level import extract_text as extract_pdf_text
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import email
from pathlib import Path

def extract_text_from_pdf(path: str) -> str:
    return extract_pdf_text(path)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_email(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        msg = email.message_from_file(f)
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_payload(decode=True).decode(errors="ignore"))
        return "\n".join(parts)
    else:
        return msg.get_payload(decode=True).decode(errors="ignore")

def extract_text(path: str) -> str:
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
        raise ValueError(f"Formato no soportado: {ext}")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]

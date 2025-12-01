# app/utils/chunker.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
):
    """
    Chunker basado en RecursiveCharacterTextSplitter.
    Mucho más robusto para textos reales (PDFs, contratos, emails, facturas).

    - chunk_size: tamaño máximo de cada chunk (caracteres)
    - chunk_overlap: cantidad de traslape entre chunks
    """

    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",     # separa por párrafos
            "\n",       # luego por líneas
            ". ",       # luego por oraciones
            " ",        # luego por palabras
            ""          # y finalmente por caracteres
        ]
    )

    chunks = splitter.split_text(text)

    # Limpieza suave
    chunks = [c.strip() for c in chunks if c.strip()]

    return chunks

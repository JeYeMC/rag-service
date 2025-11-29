# scripts/test_ingest.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.rag.ingestion import ingest_file_to_pinecone

def main():
    path = input("Ruta del PDF a indexar: ").strip()
    source = input("Nombre del origen (opcional): ").strip() or "manual-test"

    print("📄 Iniciando ingesta...")
    res = ingest_file_to_pinecone(path, source_name=source)

    print("\n=== RESULTADO ===")
    print(res)

if __name__ == "__main__":
    main()

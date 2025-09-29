import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import embed_texts
from pinecone_client import query_index

query = "¿Qué cláusulas tiene el contrato?"
vec = embed_texts([query])[0]

res = query_index("rag-index", vec, top_k=5)

for match in res["matches"]:
    print("Score:", match["score"])
    print("Chunk #:", match["metadata"].get("chunk_index"))
    print("Texto:", match["metadata"]["text_excerpt"])
    print("---")


# app/api/query.py

from fastapi import APIRouter
from pydantic import BaseModel
import time

from app.rag.pipeline import answer_question

router = APIRouter(prefix="/query", tags=["Consulta RAG"])

class QueryRequest(BaseModel):
    query: str
    doc_type: str | None = None
    provider: str = "openai"   # openai | hf

@router.post("/")
async def query_rag(q: QueryRequest):

    start = time.time()

    # Llamada correcta al pipeline (usa question=)
    result = answer_question(
        question=q.query,       # <-- CORRECTO
        top_k=15,
        doc_type=q.doc_type,
        provider=q.provider
    )

    elapsed = round(time.time() - start, 2)

    return {
        "query": q.query,
        "provider": q.provider,
        "doc_type": result["doc_type"],
        "answer": result["answer"],
        "sources": result["sources"],
        "compressed_context": result["compressed_context"],
        "elapsed_seconds": elapsed
    }

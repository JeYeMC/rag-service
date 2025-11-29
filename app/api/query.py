# app/api/query.py

from fastapi import APIRouter
from pydantic import BaseModel
import time

from app.rag.pipeline import answer_question

router = APIRouter(prefix="/query", tags=["Consulta RAG"])

class QueryRequest(BaseModel):
    query: str
    doc_type: str | None = None

@router.post("/")
async def query_rag(q: QueryRequest):
    """
    Procesa una consulta usando RAG:
      - retrieve
      - rerank
      - compress
      - prompt adaptado al doc_type
      - LLM
    """
    start = time.time()

    result = answer_question(q.query, top_k=15, doc_type=q.doc_type)

    elapsed = round(time.time() - start, 2)
    return {
        "query": q.query,
        "doc_type": result["doc_type"],
        "answer": result["answer"],
        "sources": result["sources"],
        "compressed_context": result["compressed_context"],
        "elapsed_seconds": elapsed
    }

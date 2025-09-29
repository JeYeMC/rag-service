import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

from rag_pipeline import ingest_file_to_pinecone, answer_question
from ingest_utils import extract_text, chunk_text

load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Service - CRM Project")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...), source_name: str = Form("upload")):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    result = ingest_file_to_pinecone(str(dest), source_name=source_name)
    return result

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(q: Question):
    result = answer_question(q.query, top_k=3)
    return result

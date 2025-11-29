# app/api/feedback.py

from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
import json
import time

router = APIRouter(prefix="/feedback", tags=["Feedback"])

FEEDBACK_LOG = Path("storages/feedback_log.json")
FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)

class Feedback(BaseModel):
    question: str
    answer: str
    correct: bool
    comment: str | None = None
    doc_type: str | None = None

@router.post("/")
async def save_feedback(data: Feedback):
    """Guarda feedback en un archivo JSON para futuras mejoras."""
    entry = data.dict()
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    previous = []
    if FEEDBACK_LOG.exists():
        try:
            previous = json.loads(FEEDBACK_LOG.read_text())
        except:
            previous = []

    previous.append(entry)
    FEEDBACK_LOG.write_text(json.dumps(previous, indent=2, ensure_ascii=False))

    return {"status": "feedback_saved", "count": len(previous)}

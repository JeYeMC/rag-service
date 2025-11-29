# tests/test_api.py

import os
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload():
    pdf_path = "data/uploads/test_sample.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip("No test PDF available")

    with open(pdf_path, "rb") as f:
        resp = client.post(
            "/upload",
            files={"file": ("test.pdf", f, "application/pdf")},
            data={"source_name": "pytest"},
        )

    assert resp.status_code == 200
    assert "upload_status" in resp.json()


def test_query():
    resp = client.post(
        "/query",
        json={"query": "¿De qué trata el documento?"}
    )
    assert resp.status_code == 200
    assert "response" in resp.json()

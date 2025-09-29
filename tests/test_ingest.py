from rag_service.ingest_utils import chunk_text

def test_chunking():
    text = "Lorem ipsum dolor sit amet. " * 50
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)

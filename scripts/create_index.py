import os
from dotenv import load_dotenv
import pinecone

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-index")
DIM = 384  # Modelo MiniLM

pinecone.init(api_key=API_KEY, environment=ENV)

if INDEX_NAME in pinecone.list_indexes():
    print("ℹ️ Index already exists:", INDEX_NAME)
else:
    pinecone.create_index(name=INDEX_NAME, dimension=DIM, metric="cosine")
    print("✅ Index created:", INDEX_NAME)

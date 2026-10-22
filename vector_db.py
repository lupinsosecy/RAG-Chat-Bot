import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

COLLECTION_NAME = "rag_documents"

def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "./qdrant_storage")
    api_key = os.getenv("QDRANT_API_KEY", None)

    if url.startswith("http"):
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(path=url)

def init_collection():
    client = get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    # If it already exists with OpenAI's 1536 dimension, recreate it for 768
    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=768,  # Gemini text-embedding-004 dimension
            distance=models.Distance.COSINE
        ),
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully with 768 dimensions.")

if __name__ == "__main__":
    init_collection()
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
import inngest
import inngest.fast_api
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from pydantic import BaseModel
from qdrant_client.http.models import PointStruct

from vector_db import COLLECTION_NAME, get_qdrant_client

load_dotenv()

# Initialize Gemini Client
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# Helper for embedding generation
def get_embedding(text_or_texts):
    if isinstance(text_or_texts, str):
        contents = [text_or_texts]
    else:
        contents = text_or_texts

    response = ai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=contents,
        config={"output_dimensionality": 768},
    )
    return [e.values for e in response.embeddings]


# ---------------------------------------------------------------------------
# 1. Ingestion Pipeline
# ---------------------------------------------------------------------------
@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag.ingest.pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load_documents():
        pdf_path = ctx.event.data.get("pdf_path", "./data/sample.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)

        return [
            {
                "id": node.node_id,
                "text": node.get_content(),
                "metadata": node.metadata,
            }
            for node in nodes
        ]

    chunks = await ctx.step.run("load-and-chunk-pdf", _load_documents)

    def _index_chunks():
        if not chunks:
            return {"indexed": 0}

        qdrant = get_qdrant_client()
        texts = [chunk["text"] for chunk in chunks]

        # Embed all chunks
        embeddings = get_embedding(texts)

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={
                    "text": chunks[i]["text"],
                    "metadata": chunks[i]["metadata"],
                },
            )
            for i in range(len(chunks))
        ]

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return {"indexed": len(points)}

    result = await ctx.step.run("embed-and-upsert-qdrant", _index_chunks)
    return result


# ---------------------------------------------------------------------------
# 2. Synchronous Query Endpoint for Streamlit Chat
# ---------------------------------------------------------------------------
app = FastAPI(title="Production RAG Agent")


class DirectQueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/ask")
async def ask_rag(req: DirectQueryRequest):
    try:
        qdrant = get_qdrant_client()

        # 1. Embed query
        embeddings = get_embedding(req.query)
        query_vector = embeddings[0]

        # 2. Modern Qdrant search
        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=req.top_k,
        )

        contexts = [
            point.payload.get("text", "")
            for point in search_response.points
            if point.payload and "text" in point.payload
        ]

        if not contexts:
            return {
                "answer": "No relevant documents found. Please upload a PDF first!",
                "sources": [],
            }

        # 3. Prompt construction
        context_str = "\n\n---\n\n".join(contexts)
        prompt = (
            f"You are a helpful AI assistant. Answer the user's question based on the provided context.\n"
            f"- Account for slight typos, transposed names (e.g., 'MindClip' vs 'ClipMind'), or alternate phrasing.\n"
            f"- If the answer can reasonably be inferred or clarified from the context, provide a direct and helpful explanation.\n"
            f"- Only if the context contains zero relevant information should you state that the answer was not found.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question:\n{req.query}"
        )

        # 4. Generate answer with automatic fallback for 503 spikes
        models_to_try = [
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-2.0-flash",
        ]
        
        last_error = None
        for model_id in models_to_try:
            try:
                response = ai_client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
                return {"answer": response.text, "sources": contexts}
            except Exception as e:
                last_error = e
                continue

        raise last_error

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"Error: {str(e)}", "sources": []}
    
        # 3. Generate answer with Gemini
        context_str = "\n\n---\n\n".join(contexts)
        prompt = (
            f"You are a helpful AI assistant. Answer the user's question using ONLY "
            f"the provided context. If the answer is not present in the context, say "
            f"'I could not find the answer in the uploaded documents.'\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question:\n{req.query}"
        )

        response = ai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )

        return {"answer": response.text, "sources": contexts}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"Error: {str(e)}", "sources": []}


# Register Inngest functions
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf],
)
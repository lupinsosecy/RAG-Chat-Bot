import os
import requests
import streamlit as st

FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Production RAG Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Production RAG Assistant")
st.caption("Powered by Inngest, Qdrant, LlamaIndex & Google Gemini")

# Sidebar: Document Ingestion
with st.sidebar:
    st.header("📄 Ingest Documents")
    uploaded_file = st.file_uploader("Upload a PDF to index", type=["pdf"])

    if uploaded_file and st.button("Trigger Ingestion Pipeline"):
        os.makedirs("./data", exist_ok=True)
        save_path = os.path.join("./data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Trigger Inngest event via dev server API or direct event
        try:
            inngest_url = "http://127.0.0.1:8288/e/rag_app"
            event_payload = {
                "name": "rag.ingest.pdf",
                "data": {"pdf_path": save_path},
            }
            res = requests.post(inngest_url, json=event_payload)
            if res.status_code in [200, 201]:
                st.success(f"Ingestion event triggered for '{uploaded_file.name}'!")
                st.info("Check http://127.0.0.1:8288 to watch the steps process in real time.")
            else:
                st.error("Failed to trigger Inngest event.")
        except Exception as e:
            st.error(f"Error connecting to Inngest Dev Server: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {idx}:**\n> {src}")

if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Qdrant and generating answer..."):
            try:
                resp = requests.post(
                    f"{FASTAPI_URL}/ask",
                    json={"query": prompt, "top_k": 3},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No response generated.")
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("View Retrieved Sources"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**Chunk {idx}:**\n> {src}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    st.error(f"API Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Connection error to backend: {e}")
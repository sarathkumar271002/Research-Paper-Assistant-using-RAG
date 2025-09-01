import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag.loader import load_pdfs
from rag.splitter import get_text_splitter
from rag.embed import get_embeddings
from rag.vectordb import get_or_create_chroma, add_documents, as_retriever
from rag.chain import build_qa_chain

load_dotenv()

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
st.title("📄 Research Paper Assistant using RAG")

# --- Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    collection_name = st.text_input("Collection name", value="papers")
    persist_dir = st.text_input("Chroma persist dir", value="./chroma_db")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=800, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=400, value=120, step=10)
    k_docs = st.slider("Top-k retrieved", min_value=2, max_value=10, value=4)
    use_mmr = st.checkbox("Use MMR (diverse results)", value=True)
    temperature = st.slider("LLM temperature", 0.0, 1.2, 0.2, 0.1)
    model_name = st.text_input(
        "Groq model",
        value="llama-3.1-8b-instant",
        help="Available Groq models: llama-3.1-8b-instant, llama-3.1-70b-versatile, etc."
    )

st.write("Upload PDFs, index with Chroma, and ask questions. Answers come from **Groq LLaMA 3.1** with citations.")

# --- File upload
uploaded_files = st.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = get_or_create_chroma(collection_name, persist_dir)

if uploaded_files and st.button("📥 Ingest PDFs → Chroma"):
    with st.spinner("Processing and indexing PDFs..."):
        tmp_paths = []
        for f in uploaded_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.read())
            tmp.flush()
            tmp_paths.append(tmp.name)

        docs = load_pdfs(tmp_paths)
        splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        embeddings = get_embeddings()
        add_documents(st.session_state.vectordb, chunks, embeddings)
        st.session_state.vectordb.persist()

    st.success(f"Indexed {len(chunks)} chunks into '{collection_name}'.")

# --- QA section
query = st.text_input("🔎 Ask a question", placeholder="e.g., What are the contributions of the Transformer paper?")

if query:
    with st.spinner("Retrieving and generating answer..."):
        retriever = as_retriever(st.session_state.vectordb, k=k_docs, use_mmr=use_mmr)
        chain = build_qa_chain(
            retriever=retriever,
            model_name=model_name,
            temperature=temperature,
        )
        result = chain.invoke({"question": query})

    st.subheader("✅ Answer")
    st.write(result["answer"])

    st.subheader("🔗 Sources")
    for i, ref in enumerate(result["sources"], 1):
        meta = ref.metadata
        st.write(f"{i}. {meta.get('source','unknown')} (p.{meta.get('page','?')})")

    with st.expander("🔍 Retrieved Context"):
        for i, doc in enumerate(result["context"], 1):
            st.markdown(f"**Chunk {i}** — `{doc.metadata.get('source','?')}` p.{doc.metadata.get('page','?')}\n\n{doc.page_content[:800]}…")

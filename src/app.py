import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Gemini API Key (keep secret in production)
os.environ["GEMINI_API_KEY"] = "AIzaSyCm_VmsKpS1XHOp8M1NxvKyJ8c5bVIL_F8"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "crowdfunding_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME)

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

# ──────────────────────────────────────────────
# CORE FUNCTIONS
# ──────────────────────────────────────────────

def retrieve_documents(collection, embed_model, query, top_k=3):
    query_emb = embed_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({
            "title": meta.get("title", "Untitled"),
            "text": doc
        })
    return docs

def generate_answer(query, docs):
    context = "\n\n".join([d["text"] for d in docs])
    prompt = f"""
    You are an expert crowdfunding data analyst.
    Use only the CONTEXT below to answer the QUESTION accurately.
    If the context doesn’t contain enough info, clearly say so.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    Provide a concise, insightful, and data-backed answer.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="💡 CrowdIntel RAG",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 CrowdIntel RAG: Crowdfunding Insight System")
st.markdown("Ask any question about crowdfunding campaigns. The system retrieves relevant knowledge and generates intelligent insights using RAG.")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4727/4727070.png", width=90)
st.sidebar.markdown("### ⚙️ Configuration")
top_k = st.sidebar.slider("Number of documents to retrieve:", 1, 5, 3)

collection = load_chroma()
embed_model = load_embed_model()

query = st.text_input("🔍 Ask your question:")
if st.button("💬 Generate Insight", key="generate_btn"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving knowledge and generating insight..."):
            docs = retrieve_documents(collection, embed_model, query, top_k)
            answer = generate_answer(query, docs)

        st.success("✅ Insight generated!")

        st.subheader("📘 Insight Summary")
        st.write(answer)

        with st.expander("📄 Retrieved Knowledge Context"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Document {i}: {d['title']}**")
                st.write(d["text"])

st.markdown("---")
st.caption("© 2025 CrowdIntel RAG | Powered by Gemini 2.5 Flash, ChromaDB & SentenceTransformers")

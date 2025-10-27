import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

genai.configure(api_key=os.getenv("YOUR_GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "crowdfunding_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ✅ Updated Chroma client for v0.5+
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(COLLECTION_NAME)

embedder = SentenceTransformer(EMBED_MODEL)
model = genai.GenerativeModel(MODEL_NAME)

PROMPT_TEMPLATE = """You are an evidence-based assistant. Use ONLY the provided CONTEXT documents below to answer the user's question.
If the answer cannot be found or inferred from the context, say exactly: "I don't have enough information in the provided context to answer that."

CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer concisely (2–6 sentences) using facts from the context.
- Cite sources by appending [Source: filename] after factual statements.
- If info is missing, say exactly: "I don't have enough information in the provided context to answer that."
- Do not hallucinate or invent facts.

Answer:
"""

def retrieve_top_k(query, top_k=3):
    q_emb = embedder.encode(query, convert_to_numpy=True)
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "ids", "distances"]
    )
    docs = []
    if results and "documents" in results and len(results["documents"]) > 0:
        for i in range(len(results["documents"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
    return docs

def build_context_text(docs):
    parts = []
    for d in docs:
        header = f"[Source: {d['id']}] {d['metadata'].get('title','')}"
        parts.append(header + "\n" + d["text"][:4000])
    return "\n\n---\n\n".join(parts)

def call_gemini(prompt, max_output_tokens=512, temperature=0.0):
    response = model.generate_content(
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )
    return response.text.strip()

def get_insight_from_rag(query: str, top_k: int = 3):
    retrieved = retrieve_top_k(query, top_k=top_k)
    if not retrieved:
        return {"answer": "No documents found in the knowledge base.", "documents": []}
    context = build_context_text(retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    answer = call_gemini(prompt)
    return {"answer": answer, "documents": retrieved, "prompt": prompt}
print("RAG service is ready.")

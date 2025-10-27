import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# Paths and constants
DOCS_DIR = "docs"
PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "crowdfunding_docs"


def load_docs():
    """Load all text documents from docs/ folder."""
    docs = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".txt"):
            path = os.path.join(DOCS_DIR, f)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read().strip()
            title = text.splitlines()[0] if text else f
            docs.append({"id": f, "title": title, "text": text})
    return docs


def flatten_embedding(emb):
    """Convert any embedding (numpy, nested, etc.) into a flat Python list of floats."""
    arr = np.asarray(emb, dtype=float)
    if arr.ndim == 1:
        return arr.tolist()
    elif arr.ndim == 2:
        return arr.mean(axis=0).tolist()
    else:
        raise ValueError(f"Unexpected embedding shape: {arr.shape}")


def main():
    if not os.path.exists(DOCS_DIR):
        raise FileNotFoundError(f"'{DOCS_DIR}' not found")

    os.makedirs(PERSIST_DIR, exist_ok=True)

    # Setup Chroma persistent client
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Delete collection if it already exists (to ensure clean rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Create a new collection
    collection = client.create_collection(name=COLLECTION_NAME)

    # Load documents
    docs = load_docs()
    print(f"Loaded {len(docs)} documents from '{DOCS_DIR}'")

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [{"title": d["title"]} for d in docs]

    # Initialize embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding {len(texts)} documents...")
    raw_embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Flatten embeddings and verify
    embeddings = [flatten_embedding(e) for e in raw_embs]

    # ✅ Debug: print the first embedding details
    print("\n--- DEBUG EMBEDDING INFO ---")
    print(f"Type: {type(embeddings[0])}")
    print(f"Length: {len(embeddings[0])}")
    print(f"First 3 values: {embeddings[0][:3]}")
    print("----------------------------\n")

    # Add to Chroma
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metas
    )

    print(f"✅ Successfully stored {len(docs)} documents into '{COLLECTION_NAME}' at '{PERSIST_DIR}'")


if __name__ == "__main__":
    main()

import os
import fitz 
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def load_articles(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, file))
            text = "\n".join([page.get_text() for page in doc])
            docs.append(text)
    return docs

def chunk_text(texts, chunk_size=500, overlap=50):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings, chunks

def save_faiss(embeddings, chunks):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "vector.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    texts = load_articles("articles")
    chunks = chunk_text(texts)
    embeddings, chunk_texts = embed_chunks(chunks)
    save_faiss(embeddings, chunk_texts)
    print(f"Saved {len(chunks)} chunks to FAISS index.")
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load vector index and data
index = faiss.read_index("vector.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Other models can be used. 
# The first time the script is run, it will download the model.
# The second time, it will load the model from the cache.
model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# RAG (Retrieval-Augmented Generation) function
def ask_rag(query):
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k=3)
    context = "\n".join([chunks[i] for i in I[0]])
    context = context[:1500]

    print("\n[DEBUG] Retrieved context:\n", context[:500])  # Print first 500 chars

    print("\n[DEBUG] Top matched chunks:")
    for i in I[0]:
        print("----\n", chunks[i][:300])


        prompt = f"""You are a robotics assistant AI. Use the context below to answer the question. If the answer is not in the context, say you don't know.

            Context:
            {context}

            Question: {query}

            Answer:"""

    output = gen_pipeline(prompt, max_new_tokens=150)[0]["generated_text"]
    return output.split("Answer:")[-1].strip()

# Interface for the Command Line
if __name__ == "__main__":
    print("Robotics RAG Chat (type 'exit' to quit)")
    while True:
        q = input("\nYour question: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = ask_rag(q)
        print(f"\nAnswer: {answer}")

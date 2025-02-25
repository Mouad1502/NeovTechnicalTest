from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import faiss
import numpy as np
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ✅ Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ API Key is missing! Set OPENAI_API_KEY in .env file.")

app = FastAPI()

# ✅ Serve static files (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ✅ Define a request model
class ChatRequest(BaseModel):
    question: str


# ✅ Load and Chunk Documents
def load_and_chunk_documents(directory="documents"):
    documents = []
    raw_texts = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        doc_texts = loader.load()
        raw_texts.extend([doc.page_content for doc in doc_texts])

    # ✅ Chunking the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.create_documents(raw_texts)


# ✅ BM25 Keyword Search
def create_bm25_index(documents):
    tokenized_corpus = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_corpus), tokenized_corpus


# ✅ FAISS Vector Search
def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)


# ✅ Hybrid Retrieval (BM25 + FAISS)
def hybrid_retrieval(query, bm25, tokenized_corpus, vector_store, top_k=3):
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    bm25_docs = [documents[i] for i in top_bm25_indices]

    faiss_docs = vector_store.similarity_search(query, k=top_k)
    return bm25_docs + faiss_docs


# ✅ Setup QA System
def setup_qa_system():
    global documents, bm25, tokenized_corpus, vector_store
    documents = load_and_chunk_documents()
    bm25, tokenized_corpus = create_bm25_index(documents)
    vector_store = create_vector_store(documents)
    print("✅ Documents Loaded & Indexed Successfully!")


# ✅ Initialize on Startup
setup_qa_system()


# ✅ Chatbot API Endpoint
@app.post("/chat")
async def chat(query: ChatRequest):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # ✅ Retrieve relevant documents
    retrieved_docs = hybrid_retrieval(query.question, bm25, tokenized_corpus, vector_store)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # ✅ Force the model to answer **ONLY using provided context**
    prompt = f"""
    You are an AI assistant. Answer the following question **only using the provided context**.
    If the answer is not in the context, say "I don't know."

    **Context:**
    {context}

    **Question:**
    {query.question}
    """

    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return {"response": response.json()["choices"][0]["message"]["content"]}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Run server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)

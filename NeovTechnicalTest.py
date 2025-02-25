#%%
import subprocess
import sys


# Function to install missing packages
def install_packages():
    required_packages = [
        "faiss-cpu", "chromadb", "langchain", "rank_bm25", "sentence-transformers",
        "transformers", "torch", "langchain-community", "pypdf"
    ]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))  # Try to import the package
        except ImportError:
            print(f"📦 Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# ✅ Install required packages before running the script
install_packages()
#%% md
# ### 1️⃣ Import Required Libraries
# 
# 
# 
#%%
import os
import faiss
import numpy as np
import chromadb
import torch
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
#%% md
# Connection with the API
#%%
import requests

# ✅ Use OpenRouter API directly
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ✅ Replace with your actual OpenRouter API key
OPENAI_API_KEY = "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"

def ask_openai_api(question, context):
    """Send a request to OpenRouter API with explicit document context."""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # ✅ Force the model to only use the provided context
    prompt = f"""
    You are an AI assistant. Answer the following question **only using the provided context**.
    If the answer is not in the context, say "I don't know."

    **Context:**
    {context}

    **Question:**
    {question}
    """

    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None
#%% md
# ### 2️⃣ Load and Chunk Documents
# 
#%%
def load_and_chunk_documents(directory):
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

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents(raw_texts)
    return documents
#%% md
# ###3️⃣ Implement BM25 for Keyword-Based Search
#%%
def create_bm25_index(documents):
    tokenized_corpus = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus
#%% md
# ###4️⃣ Implement Hybrid Retrieval (BM25 + FAISS)
#%%
def hybrid_retrieval(query, bm25, tokenized_corpus, vector_store, top_k=3):
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    bm25_docs = [documents[i] for i in top_bm25_indices]

    faiss_docs = vector_store.similarity_search(query, k=top_k)
    return bm25_docs + faiss_docs
#%% md
# ###5️⃣ Create Vector Store for Semantic Search
#%%
def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store
#%% md
# ###6️⃣ Setup the QA System with RAG
#%%
def setup_qa_system():
    documents = load_and_chunk_documents("documents")
    bm25, tokenized_corpus = create_bm25_index(documents)
    vector_store = create_vector_store(documents)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the following context: {context}, answer the question: {question}"
    )

    # Use API instead of local model
    llm = lambda question: ask_openai_api(question)
    qa_chain = LLMChain(llm=llm, prompt=prompt_template)

    return qa_chain, bm25, tokenized_corpus, vector_store, documents
#%% md
# ###7️⃣ Query the System
#%%
def ask_question(qa_chain, bm25, tokenized_corpus, vector_store, documents, question):
    retrieved_docs = hybrid_retrieval(question, bm25, tokenized_corpus, vector_store)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    response = qa_chain.run({"context": context, "question": question})
    return response
#%% md
# ###8️⃣ Run the System and Test
#%%
if __name__ == "__main__":
    # ✅ Notify that the script is running
    print("🚀 Running Jupyter Notebook as a script...")

    # ✅ Setup the QA system (if needed)
    qa_chain, bm25, tokenized_corpus, vector_store, documents = setup_qa_system()

    while True:
        user_question = input("\n📝 Ask a question (or type 'exit' to quit): ")
        if user_question.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        # ✅ Use retrieved documents for context-based response
        answer = ask_question(qa_chain, bm25, tokenized_corpus, vector_store, documents, user_question)

        print("\n🎯 AI Response:\n", answer if answer else "❌ No response.")


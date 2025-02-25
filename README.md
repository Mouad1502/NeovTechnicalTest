# 🧠 Neov Chatbot - RAG-Based Question Answering System

## 📌 Overview
Neov Chatbot is a **Retrieval-Augmented Generation (RAG)** system designed to answer questions using **provided reference documents**. It integrates **BM25 for keyword search**, **FAISS for semantic search**, and **OpenRouter API** to generate answers.

## 🚀 Features
- ✅ Uses **BM25 + FAISS** for hybrid retrieval  
- ✅ Supports **text and PDF documents** for context-based answers  
- ✅ **FastAPI-powered API** with a simple **frontend UI**  
- ✅ **Integrated OpenRouter API** for LLM-based responses  
- ✅ **Locally tested and fully functional**  

## 👀 How It Works
1. **Document Processing**: The system loads and chunks text/PDF files.
2. **Hybrid Retrieval**:
   - **BM25** → Finds relevant documents using keyword search.
   - **FAISS** → Embeds documents and retrieves them using vector similarity.
3. **Query Handling**:
   - The chatbot searches the **most relevant context** using **BM25 + FAISS**.
   - It **sends the context & question** to OpenRouter API for response generation.
4. **Chat UI**: A simple **frontend** (HTML/CSS/JS) to interact with the chatbot.

## 📂 Project Structure
```
📁 NeovTechnicalTest
│── 📁 static                # Frontend (HTML, CSS, JS)
│   ├── index.html           # Chat UI
│   ├── style.css            # UI styling
│   ├── script.js            # Handles user input & API calls
│── 📁 documents             # Reference documents (PDFs & TXT)
│── 📄 server.py             # FastAPI backend (BM25, FAISS, API)
│── 📄 requirements.txt      # Python dependencies
│── 📄 Procfile              # Deployment instructions (if needed)
│── 📄 .env                  # Environment variables (API key)
│── 📄 README.md             # Project documentation
```

## 🛠️ Setup & Running Locally
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/neov-chatbot.git
cd neov-chatbot
```
### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```
### 3️⃣ **Set Up Environment Variables**
Create a **`.env` file** and add:
```env
OPENAI_API_KEY=your-api-key-here
```
### 4️⃣ **Run the Server**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
✅ **Go to:** `http://localhost:8000` → You should see the chat UI!

---

## 🚀 Deployment Status
**Neov Chatbot is deployable**, but **free hosting services** (e.g., Render, Railway) **don’t provide enough RAM** for FAISS + HuggingFace models.

### **What We Tried**
| Approach | Worked? | Issue |
|----------|--------|-------|
| **Local Execution** | ✅ Yes | Fully functional with enough RAM |
| **FastAPI + OpenRouter API** | ✅ Yes | Uses external LLM for inference |
| **Render** | ❌ No | Limited RAM (512MB) - FAISS crashes |
| **Railway** | ❌ No | Free plan doesn’t support high RAM usage |
| **Vercel** | ❌ No | Doesn’t support long-running Python processes |

### **Possible Solutions**
- **Use a paid hosting service** (AWS, GCP, Linode, DigitalOcean).
- **Reduce memory footprint** (optimize FAISS indexing, limit document size).
- **Use serverless OpenRouter API** instead of local embeddings.

---

## 🛠️ Technologies Used
- 🟢 **FastAPI** - Backend framework  
- 🟢 **LangChain** - Document processing & retrieval  
- 🟢 **FAISS** - Semantic search with embeddings  
- 🟢 **BM25 (Rank-BM25)** - Keyword-based search  
- 🟢 **HuggingFace Embeddings** - Vector embeddings  
- 🟢 **OpenRouter API** - LLM-powered responses  
- 🟢 **Uvicorn** - ASGI server  
- 🟢 **HTML, CSS, JS** - Frontend chat UI  

---

## 💀 Future Improvements
✅ Reduce memory usage for deployment compatibility  
✅ Improve response speed using optimized embeddings  
✅ Add **user authentication** for API security  
✅ Support **multiple LLM providers** (DeepSeek, GPT-4o, Mistral, etc.)  

---

## 🤝 Contributing
Want to improve this chatbot? Feel free to fork, modify, and submit a **pull request**! 🚀

---

## 📄 License
MIT License. Free to use and modify.

---

## 📱 Contact
For any questions, feel free to reach out!  
✉️ Email: mrmouadabbar@gmail.com  
🐙 GitHub: [Mouad1502](https://github.com/Mouad1502)
```


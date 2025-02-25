from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

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

# ✅ Chatbot API endpoint
@app.post("/chat")
async def chat(query: ChatRequest):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [{"role": "user", "content": query.question}]
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
    uvicorn.run(app, host="0.0.0.0", port=8001)

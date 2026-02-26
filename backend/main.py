import os
import json
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file")

BASE_URL = "https://api.deepseek.com/v1"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# Load Local Embedding Model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load Documents
# -----------------------------
with open("docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# -----------------------------
# Document Chunking
# -----------------------------
def chunk_text(text, chunk_size=400):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunks = []

for doc in documents:
    for chunk in chunk_text(doc["content"]):
        chunks.append({
            "title": doc["title"],
            "content": chunk,
            "embedding": None
        })

# -----------------------------
# Generate Embeddings at Startup
# -----------------------------
print("Generating document embeddings...")

for chunk in chunks:
    chunk["embedding"] = embedding_model.encode(chunk["content"])

print("Embeddings generated successfully.")

# -----------------------------
# Request Model
# -----------------------------
class ChatRequest(BaseModel):
    sessionId: str
    message: str

# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/api/chat")
def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # 1. Embed user query
        query_embedding = embedding_model.encode(request.message)

        # 2. Compute cosine similarity
        similarity_scores = []

        for chunk in chunks:
            score = cosine_similarity(
                [query_embedding],
                [chunk["embedding"]]
            )[0][0]

            similarity_scores.append((score, chunk))

        # 3. Sort by similarity
        similarity_scores.sort(reverse=True, key=lambda x: x[0])

        # 4. Retrieve top 3
        top_chunks = similarity_scores[:3]

        # 5. Similarity threshold
        if top_chunks[0][0] < 0.3:
            return {
                "reply": "I do not have enough information to answer that.",
                "tokensUsed": 0,
                "retrievedChunks": 0
            }

        # 6. Build context
        context = "\n\n".join([c[1]["content"] for c in top_chunks])

        # 7. Construct grounded prompt
        prompt = f"""
You are a grounded AI assistant.

Use ONLY the provided context to answer.
If the answer is not in the context, say:
"I do not have enough information."

Context:
{context}

Question:
{request.message}
"""

        # 8. Call DeepSeek Chat API
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "temperature": 0.2,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=60
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=response.text)

        result = response.json()

        reply = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)

        return {
            "reply": reply,
            "tokensUsed": tokens_used,
            "retrievedChunks": len(top_chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
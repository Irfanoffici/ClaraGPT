# /api/main.py
from fastapi import FastAPI, Query
from mangum import Mangum
from utils.fetch_data import fetch_medical_data
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import google.generativeai as genai
import os

app = FastAPI(title="ClaraGPT")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],  # replace with your frontend URL
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 🔹 Gemini API (get key from Vercel environment variables)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# 🔹 Lazy RAG DB initialization
_db = None

def get_rag_db():
    global _db
    if _db is None:
        print("🔄 Fetching medical data and building RAG DB...")
        medical_text = fetch_medical_data()
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        docs = splitter.split_documents([Document(page_content=medical_text)])
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _db = Chroma.from_documents(docs, embeddings)
        print("✅ RAG DB ready")
    return _db

@app.get("/")
def root():
    return {"message": "ClaraGPT is online. Use /ask?q=Your+question"}

@app.get("/ask")
def ask(q: str = Query(..., description="Medical question")):
    db = get_rag_db()
    retrieved = db.similarity_search(q, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved])

    prompt = f"""
You are ClaraGPT, an AI medical assistant.
Answer accurately using the context below and cite sources.

Question: {q}

Context:
{context}
"""
    response = model.generate_content(prompt)
    return {
        "question": q,
        "answer": response.text,
        "citations": [f"Source {i+1}" for i in range(len(retrieved))]
    }

# Wrap FastAPI app for Vercel
handler = Mangum(app)

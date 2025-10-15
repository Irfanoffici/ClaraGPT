# ClaraGPT Backend using Gemini + FastAPI + Chroma
from fastapi import FastAPI, Query
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import google.generativeai as genai
import os

app = FastAPI(title="ClaraGPT API", version="1.0")

# --- 1. Configure Gemini ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyAqAp5_60wxyspiLM0XnX3LBj6hY3GBBHc"  # put your key here
genai.configure(api_key=os.environ["AIzaSyAqAp5_60wxyspiLM0XnX3LBj6hY3GBBHc"])
model = genai.GenerativeModel("gemini-1.5-flash")

# --- 2. Load sample dataset (replace with your own) ---
with open("data/medical.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents([Document(page_content=text)])

# --- 3. Build vector DB (Chroma, in-memory) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

# --- 4. Query handler ---
@app.get("/ask")
def ask(q: str = Query(...)):
    # Retrieve top 3 relevant chunks
    retrieved = db.similarity_search(q, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved])
    citations = [f"Source {i+1}" for i in range(len(retrieved))]

    # Send to Gemini
    prompt = f"""You are ClaraGPT, a medical assistant.
Use the following context to answer clearly and cite sources.
Question: {q}

Context:
{context}

Answer with numbered citations."""
    response = model.generate_content(prompt)
    answer = response.text

    return {
        "question": q,
        "answer": answer,
        "sources": citations
    }

# --- 5. Root ---
@app.get("/")
def home():
    return {"message": "ClaraGPT API is running!"}

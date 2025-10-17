from fastapi import FastAPI, Query
from utils.fetch_data import fetch_medical_data
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import google.generativeai as genai
import os

app = FastAPI(title="ClaraGPT API")

# 🔹 Configure Gemini
# --- 1. Configure Gemini ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyAqAp5_60wxyspiLM0XnX3LBj6hY3GBBHc"  # put your key here
genai.configure(api_key=os.environ["AIzaSyAqAp5_60wxyspiLM0XnX3LBj6hY3GBBHc"])
model = genai.GenerativeModel("gemini-1.5-flash")

# 🔹 Fetch + Index Data
print("Fetching medical data...")
medical_text = fetch_medical_data()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents([Document(page_content=medical_text)])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)
print("✅ Vector DB built successfully")

# 🔹 Chat Endpoint
@app.get("/ask")
def ask(q: str = Query(..., description="User question")):
    retrieved = db.similarity_search(q, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved])

    prompt = f"""
You are ClaraGPT, an AI medical assistant.
Use the context to answer factually, and cite sources.

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

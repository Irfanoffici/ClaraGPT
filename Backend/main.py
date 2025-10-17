from fastapi import FastAPI, Query
from utils.fetch_data import fetch_medical_data
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import google.generativeai as genai
import os

app = FastAPI(title="ClaraGPT v2")

# 🔹 Configure Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyAqAp5_60wxyspiLM0XnX3LBj6hY3GBBHc"  # put your key here
genai.configure(api_key=os.environ["AIzaSyAqAp5_60wxyspiLM0XnX3LBj6hY3GBBHc"])
model = genai.GenerativeModel("gemini-1.5-flash")

# 🔹 Fetch + index data
print("🔄 Fetching and processing medical data...")
medical_text = fetch_medical_data()

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
docs = splitter.split_documents([Document(page_content=medical_text)])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)
print("✅ RAG knowledge base built successfully")

@app.get("/")
def root():
    return {"message": "ClaraGPT v2 is online. Use /ask?q=Your+question"}

@app.get("/ask")
def ask(q: str = Query(..., description="Your medical question")):
    retrieved = db.similarity_search(q, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved])

    prompt = f"""
You are ClaraGPT, an AI medical assistant.
Answer accurately using the context below, and cite your sources at the end.
Avoid hallucination, stick to data.

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

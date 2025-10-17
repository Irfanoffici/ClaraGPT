# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import groq
import os
from typing import List, Dict
import uvicorn
from datetime import datetime

# Initialize the app with ClaraGPT branding
app = FastAPI(
    title="ClaraGPT Medical Assistant",
    description="A RAG-powered medical chatbot providing accurate, citation-backed health information",
    version="1.0.0",
    docs_url="/",
    redoc_url="/docs"
)

# Data models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    response_time: float

print("🩺 Starting ClaraGPT Medical Assistant...")
print("🔑 Groq API Key status: ✅ Valid" if os.getenv("GROQ_API_KEY") else "🔑 Groq API Key status: ❌ Missing")

# Initialize components
print("📥 Loading embedding model...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    exit(1)

# Initialize ChromaDB
print("🗃️ Setting up vector database...")
try:
    client = chromadb.Client()
    collection = client.create_collection(name="clara_medical_knowledge")
    print("✅ Vector database ready!")
except Exception as e:
    print(f"❌ Error setting up vector database: {e}")
    exit(1)

# Initialize Groq client with YOUR API KEY
groq_client = None

def load_medical_data():
    """Load medical data into the vector database"""
    print("📚 Loading medical knowledge for ClaraGPT...")
    try:
        with open('data/medical_data.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split into chunks (simple paragraph splitting)
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        print(f"📖 Found {len(chunks)} medical knowledge chunks")
        
        # Add to vector database
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": "ClaraGPT Medical Database", 
                    "chunk_id": i,
                    "timestamp": str(datetime.now())
                }],
                ids=[f"clara_chunk_{i}"]
            )
        
        print(f"✅ Loaded {len(chunks)} medical knowledge chunks into ClaraGPT database")
        return len(chunks)
        
    except Exception as e:
        print(f"❌ Error loading medical data: {e}")
        print("🔄 Creating default medical data for ClaraGPT...")
        
        default_chunks = [
            "Hypertension (high blood pressure): A condition where blood pressure in arteries is persistently elevated. Normal BP is below 120/80 mmHg. Symptoms may include headaches and shortness of breath.",
            "Diabetes Type 2: A metabolic disorder with high blood sugar and insulin resistance. Symptoms include increased thirst, frequent urination, and blurred vision. Managed through diet and medication.",
            "Influenza (Flu): Viral infection attacking respiratory system. Symptoms: fever, cough, sore throat, muscle aches. Prevention includes annual vaccination.",
            "Asthma: Condition where airways become inflamed and narrowed. Symptoms include wheezing, coughing, chest tightness. Triggers include allergens and exercise.",
            "Migraine: Neurological condition with intense headaches. Symptoms: throbbing pain, light sensitivity, nausea. Treatment includes pain relievers."
        ]
        
        for i, chunk in enumerate(default_chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": "ClaraGPT Default Medical Library", "chunk_id": i}],
                ids=[f"clara_default_{i}"]
            )
        
        print("✅ Loaded ClaraGPT default medical data")
        return len(default_chunks)

def search_medical_knowledge(question: str, n_results: int = 3):
    """Search for relevant medical information"""
    try:
        question_embedding = embedding_model.encode(question).tolist()
        
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )
        
        print(f"🔍 ClaraGPT found {len(results['documents'][0])} relevant chunks")
        return results
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return {'documents': [[]], 'metadatas': [[]]}

def generate_medical_answer(question: str, context_chunks: List[str]):
    """Generate answer using RAG pattern with ClaraGPT personality"""
    
    context = "\n\n".join([f"[Medical Reference {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""You are ClaraGPT, a compassionate and accurate medical assistant. Answer the user's question using ONLY the medical context provided below.

MEDICAL KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

AS CLARAGPT, FOLLOW THESE RULES:
1. Use ONLY the information from the medical context above - never invent or assume
2. If context doesn't contain relevant information, say: "I don't have enough specific medical information to answer this question accurately. Please consult a healthcare professional."
3. Cite your sources clearly using [Medical Reference 1], [Medical Reference 2], etc.
4. Be empathetic, clear, and professional
5. Include important disclaimers: "This is for informational purposes only. Always consult healthcare professionals for medical advice."
6. Structure your answer clearly with proper formatting

CLARAGPT'S ANSWER:"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama2-70b-4096",
            temperature=0.1,
            max_tokens=600
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"🩺 I apologize, but ClaraGPT is experiencing technical difficulties. Please try again shortly. Error: {str(e)}"

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Welcome to ClaraGPT Medical Assistant",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "RAG-powered medical responses",
            "Citation-backed answers", 
            "Real-time health information",
            "Built for Hack-A-Cure Hackathon"
        ],
        "endpoints": {
            "health_check": "/health",
            "ask_question": "/ask (POST)"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "ClaraGPT Medical Assistant",
        "timestamp": str(datetime.now()),
        "rag_system": "active"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Main endpoint for medical questions - ClaraGPT"""
    import time
    start_time = time.time()
    
    print(f"❓ ClaraGPT received: {request.question}")
    
    try:
        if not groq_client:
            return QuestionResponse(
                answer="🔧 ClaraGPT is currently initializing. Please try again in a moment.",
                sources=[],
                confidence=0.0,
                response_time=0.0
            )
        
        # Search for relevant medical information
        search_results = search_medical_knowledge(request.question)
        
        if not search_results['documents'] or not search_results['documents'][0]:
            return QuestionResponse(
                answer="🔍 I couldn't find specific medical information to answer your question accurately. Please try rephrasing or ask about common medical conditions like hypertension, diabetes, or flu.",
                sources=[],
                confidence=0.0,
                response_time=round(time.time() - start_time, 2)
            )
        
        context_chunks = search_results['documents'][0]
        sources = [f"ClaraGPT Medical Reference {i+1}" for i in range(len(context_chunks))]
        
        # Generate answer using RAG
        answer = generate_medical_answer(request.question, context_chunks)
        
        # Calculate confidence score
        confidence = min(len(context_chunks) / 3, 1.0)
        response_time = round(time.time() - start_time, 2)
        
        print(f"✅ ClaraGPT answered in {response_time}s with confidence: {confidence}")
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            confidence=round(confidence, 2),
            response_time=response_time
        )
        
    except Exception as e:
        print(f"❌ ClaraGPT error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"ClaraGPT encountered an error: {str(e)}"
        )

# Initialize everything on startup
@app.on_event("startup")
async def startup_event():
    print("🔧 Initializing ClaraGPT Medical System...")
    
    # Initialize Groq client with YOUR API KEY
    global groq_client
    groq_api_key = os.getenv("GROQ_API_KEY", "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")
    
    try:
        groq_client = groq.Groq(api_key=groq_api_key)
        print("✅ Groq client initialized for ClaraGPT!")
        
        # Test the connection
        test_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'ClaraGPT is ready'"}],
            model="llama2-70b-4096",
            max_tokens=10
        )
        print(f"✅ Groq connection test: {test_completion.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Error initializing Groq client: {e}")
        print("⚠️  ClaraGPT will start but cannot answer questions without Groq API")
    
    # Load medical data
    chunks_loaded = load_medical_data()
    print(f"🎉 ClaraGPT Medical Assistant ready! Loaded {chunks_loaded} knowledge chunks.")
    print(f"🌐 API available at: http://localhost:8000")
    print(f"📚 Documentation at: http://localhost:8000/")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

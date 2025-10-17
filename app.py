# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import groq
import os
import uvicorn
from typing import List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI with ClaraGPT branding
app = FastAPI(
    title="ClaraGPT Medical Assistant",
    description="A secure RAG-powered medical chatbot providing accurate, citation-backed health information",
    version="2.0.0",
    docs_url="/",
    redoc_url="/docs",
    openapi_tags=[
        {
            "name": "Medical",
            "description": "Medical question answering endpoints"
        },
        {
            "name": "System",
            "description": "System health and monitoring"
        }
    ]
)

# Security
security = HTTPBearer()
API_TOKENS = set()

# Data models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    response_time: float
    model_used: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    groq_status: str
    vector_db_status: str

class SystemStatus(BaseModel):
    groq_connected: bool
    vector_db_loaded: bool
    medical_chunks_count: int
    startup_time: str

# Global variables
embedding_model = None
collection = None
groq_client = None
system_startup_time = datetime.now()

print("🩺 Starting ClaraGPT Medical Assistant...")
print("🔒 Security: API Token authentication enabled")

def initialize_secure_tokens():
    """Initialize or load API tokens"""
    global API_TOKENS
    # You can pre-share tokens or generate them
    default_token = os.getenv("CLARA_API_TOKEN", "clara-default-token-2024")
    API_TOKENS.add(default_token)
    print(f"🔑 API Tokens initialized: {len(API_TOKENS)} tokens loaded")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    if token not in API_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token. Please check your authorization header."
        )
    return token

def initialize_groq_client():
    """Safely initialize Groq client with error handling"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment variables!")
        print("💡 Please set GROQ_API_KEY in your .env file or environment")
        return None
    
    if groq_api_key.startswith("gsk_") and len(groq_api_key) > 30:
        try:
            client = groq.Groq(api_key=groq_api_key)
            # Test connection
            test_response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Say 'Connected'"}],
                model="llama2-70b-4096",
                max_tokens=10
            )
            print("✅ Groq client initialized and connected successfully!")
            return client
        except Exception as e:
            print(f"❌ Failed to initialize Groq client: {e}")
            return None
    else:
        print("❌ Invalid Groq API key format")
        return None

def initialize_embedding_model():
    """Initialize the embedding model"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return None

def initialize_vector_db():
    """Initialize ChromaDB vector database with persistence"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="clara_medical_knowledge")
        print("✅ Vector database initialized successfully!")
        return collection
    except Exception as e:
        print(f"❌ Error initializing vector database: {e}")
        return None

def load_medical_data():
    """Load medical data into vector database with enhanced content"""
    if not collection:
        return 0
        
    print("📚 Loading medical knowledge base...")
    
    medical_content = """
CARDIOVASCULAR HEALTH:
Hypertension (High Blood Pressure): A condition where blood pressure in arteries is persistently elevated. Normal: <120/80 mmHg. Stage 1: 130-139/80-89 mmHg. Symptoms: headaches, shortness of breath, nosebleeds (in severe cases). Risk factors: family history, obesity, high salt intake, stress. Treatment: lifestyle changes (diet, exercise), medications (ACE inhibitors, diuretics, beta-blockers). Regular monitoring recommended.

DIABETES:
Diabetes Type 2: Metabolic disorder with high blood sugar and insulin resistance. Symptoms: increased thirst (polydipsia), frequent urination (polyuria), increased hunger (polyphagia), blurred vision, slow healing. Complications: cardiovascular disease, neuropathy, kidney damage, eye problems. Management: diet control, regular exercise, blood sugar monitoring, medications (metformin, insulin). Prevention: healthy weight, balanced diet, physical activity.

RESPIRATORY CONDITIONS:
Influenza (Flu): Viral respiratory infection. Symptoms: fever, cough, sore throat, runny nose, muscle aches, fatigue, headaches. Prevention: annual vaccination, hand hygiene, avoiding sick individuals. Treatment: rest, fluids, antiviral medications (oseltamivir). Complications: pneumonia, bronchitis. High-risk groups: elderly, children, immunocompromised.

Asthma: Chronic inflammatory airway disease. Symptoms: wheezing, coughing, chest tightness, shortness of breath. Triggers: allergens, exercise, cold air, stress, respiratory infections. Treatment: quick-relief inhalers (bronchodilators), long-term control medications (corticosteroids). Management: avoid triggers, action plan, regular check-ups.

NEUROLOGICAL CONDITIONS:
Migraine: Neurological disorder with recurrent headaches. Symptoms: throbbing pain (often one-sided), sensitivity to light/sound, nausea, vomiting. Triggers: stress, hormonal changes, certain foods, sleep changes. Treatment: pain relievers, triptans, preventive medications. Management: identify triggers, lifestyle modifications, stress management.

COMMON CONDITIONS:
Common Cold: Viral upper respiratory infection. Symptoms: runny nose, sore throat, cough, congestion, mild fever. Prevention: hand washing, avoid touching face. Treatment: rest, fluids, over-the-counter symptom relief. Duration: 7-10 days.

Gastroenteritis: Inflammation of stomach and intestines. Symptoms: diarrhea, vomiting, abdominal pain, fever. Causes: viruses, bacteria, parasites. Treatment: hydration, rest, bland diet. Prevention: food safety, hand hygiene.

MENTAL HEALTH:
Anxiety Disorders: Excessive worry and fear. Symptoms: restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep disturbances. Treatment: therapy (CBT), medications, lifestyle changes. Management: stress reduction, regular exercise, healthy sleep.

Depression: Mood disorder with persistent sadness. Symptoms: depressed mood, loss of interest, changes in appetite/sleep, fatigue, feelings of worthlessness. Treatment: psychotherapy, antidepressants, lifestyle changes. Support: social connections, routine, physical activity.
"""
    
    try:
        # Split into meaningful chunks
        chunks = []
        current_section = ""
        
        for line in medical_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.isupper() and line.endswith(':'):
                # New section
                if current_section:
                    chunks.append(current_section.strip())
                current_section = line + " "
            else:
                current_section += line + " "
        
        if current_section:
            chunks.append(current_section.strip())
        
        # Clear existing data and add new
        collection.delete(where={})
        
        # Add to vector database
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": "ClaraGPT Medical Database",
                    "chunk_id": i,
                    "category": chunk.split(':')[0] if ':' in chunk else "General",
                    "timestamp": str(datetime.now())
                }],
                ids=[f"medical_chunk_{i}"]
            )
        
        print(f"✅ Loaded {len(chunks)} medical knowledge chunks")
        return len(chunks)
        
    except Exception as e:
        print(f"❌ Error loading medical data: {e}")
        return 0

def search_medical_knowledge(question: str, n_results: int = 4):
    """Search for relevant medical information"""
    try:
        question_embedding = embedding_model.encode(question).tolist()
        
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return {'documents': [[]], 'metadatas': [[]]}

def generate_medical_answer(question: str, context_chunks: List[str]):
    """Generate answer using RAG with enhanced medical safety"""
    
    context = "\n\n".join([f"MEDICAL_REFERENCE_{i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""You are ClaraGPT, a compassionate and precise medical AI assistant. Your purpose is to provide accurate, evidence-based health information while emphasizing the crucial importance of professional medical consultation.

MEDICAL KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

CRITICAL MEDICAL GUIDELINES:
1. ANSWER STRICTLY based on the provided medical knowledge base above
2. If information is insufficient or unclear, state: "I don't have comprehensive information to answer this medical question accurately. Please consult a healthcare provider for proper diagnosis and treatment."
3. CITE SOURCES clearly using [MEDICAL_REFERENCE_1], [MEDICAL_REFERENCE_2] format
4. STRUCTURE YOUR RESPONSE:
   - Direct answer based on available information
   - Key symptoms/characteristics (if applicable)
   - General management approaches (if mentioned in sources)
   - When to seek medical attention
5. Include this DISCLAIMER prominently: "IMPORTANT: I am an AI assistant providing general health information. This is not medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns."

CLARAGPT MEDICAL RESPONSE:"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama2-70b-4096",
            temperature=0.1,
            max_tokens=800
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"🔧 ClaraGPT is currently experiencing technical difficulties. Please try again in a moment. Error: {str(e)}"

# API Routes
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "ClaraGPT Medical Assistant API", "status": "operational"}

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check"""
    groq_status = "connected" if groq_client else "disconnected"
    vector_status = "loaded" if collection else "unavailable"
    
    return HealthResponse(
        status="healthy",
        timestamp=str(datetime.now()),
        version="2.0.0",
        groq_status=groq_status,
        vector_db_status=vector_status
    )

@app.get("/system/status", response_model=SystemStatus, tags=["System"])
async def system_status(_: str = Depends(verify_token)):
    """Detailed system status (protected)"""
    return SystemStatus(
        groq_connected=groq_client is not None,
        vector_db_loaded=collection is not None,
        medical_chunks_count=collection.count() if collection else 0,
        startup_time=str(system_startup_time)
    )

@app.post("/ask", response_model=QuestionResponse, tags=["Medical"])
async def ask_question(
    request: QuestionRequest, 
    _: str = Depends(verify_token)
):
    """Ask ClaraGPT a medical question (protected endpoint)"""
    import time
    start_time = time.time()
    
    print(f"❓ Medical question received: {request.question[:100]}...")
    
    # Check system readiness
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="ClaraGPT medical service is currently unavailable. Groq API not configured."
        )
    
    if not collection:
        raise HTTPException(
            status_code=503,
            detail="Medical knowledge base not loaded. Please try again later."
        )
    
    try:
        # Search for relevant medical information
        search_results = search_medical_knowledge(request.question)
        
        if not search_results['documents'] or not search_results['documents'][0]:
            return QuestionResponse(
                answer="🔍 I couldn't find specific medical information to answer your question accurately. Please try rephrasing or ask about common medical conditions like hypertension, diabetes, asthma, or influenza.",
                sources=[],
                confidence=0.0,
                response_time=round(time.time() - start_time, 2),
                model_used="none"
            )
        
        context_chunks = search_results['documents'][0]
        sources = [f"Medical Reference {i+1}" for i in range(len(context_chunks))]
        
        # Generate answer using RAG
        answer = generate_medical_answer(request.question, context_chunks)
        
        # Calculate metrics
        confidence = min(len(context_chunks) / 4, 1.0)
        response_time = round(time.time() - start_time, 2)
        
        print(f"✅ Medical answer generated in {response_time}s")
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            confidence=round(confidence, 2),
            response_time=response_time,
            model_used="llama2-70b-4096"
        )
        
    except Exception as e:
        print(f"❌ Error processing medical question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ClaraGPT encountered an error while processing your question: {str(e)}"
        )

@app.post("/auth/test", tags=["System"])
async def test_auth(_: str = Depends(verify_token)):
    """Test authentication (protected)"""
    return {"message": "Authentication successful", "status": "valid"}

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global embedding_model, collection, groq_client
    
    print("🔧 Initializing ClaraGPT Medical System...")
    
    # Initialize security
    initialize_secure_tokens()
    
    # Initialize components in order
    embedding_model = initialize_embedding_model()
    collection = initialize_vector_db()
    groq_client = initialize_groq_client()
    
    # Load medical data
    if collection:
        chunks_loaded = load_medical_data()
        print(f"📊 Medical system loaded: {chunks_loaded} knowledge chunks")
    else:
        print("⚠️  Cannot load medical data - vector database unavailable")
    
    print("🎉 ClaraGPT Medical Assistant is ready!")
    print("🔒 Protected endpoints require Authorization: Bearer <token>")
    print("🌐 API documentation: http://localhost:8000/")

# Error handlers
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error - ClaraGPT service unavailable"}
    )

@app.exception_handler(429)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded - please try again later"}
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

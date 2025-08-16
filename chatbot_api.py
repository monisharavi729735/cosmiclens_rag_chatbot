from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import os
import requests
from dotenv import load_dotenv
from difflib import SequenceMatcher
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Load .env
load_dotenv()

app = FastAPI()

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class QueryRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []

# Static fallback
STATIC_FALLBACK = (
    "I'm not confident enough to answer that accurately right now.\n"
    "Please refer to other relevant resources for clarification."
)

# Identity detection
def is_identity_question(message):
    identity_keywords = [
        "who are you", "what are you", "your name",
        "who made you", "who created you",
        "are you from openai", "are you from mistral", "are you a bot",
        "who built you", "what's your name"
    ]
    lower_msg = message.lower()
    return any(kw in lower_msg for kw in identity_keywords)

# Hallucination detection
def is_irrelevant_or_hallucinated(question, answer):
    vague_phrases = [
        "i don't know", "i'm not sure", "i cannot answer",
        "unable to", "as an ai", "not trained", "don't have that information"
    ]
    vague = any(phrase in answer.lower() for phrase in vague_phrases)
    similarity = SequenceMatcher(None, question.lower(), answer.lower()).ratio()
    short = len(answer.split()) < 10
    return (vague and similarity < 0.3) or (short and similarity < 0.1)

# Groq (Mistral) fallback
def try_mistral(prompt):
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return None

        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }

        resp = requests.post(groq_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Mistral fallback failed: {e}")
        return None

# Setup models + embeddings
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found.")

llm = ChatGoogleGenerativeAI(
    temperature=0.5,
    model="gemini-1.5-flash",
    google_api_key=google_api_key
)

'''
# using this in an actual prod environment but I can't afford to do so now due to limited ram in free tier
vector_store = Chroma(
    collection_name="physics_textbook_collection",
    persist_directory="chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={'k': 5})
'''

class DemoRetriever:
    def __init__(self):
        self.docs = [
            "Newton's three laws explain how forces affect motion.",
            "Energy can neither be created nor destroyed — only transformed.",
            "Light behaves as both a wave and a particle depending on observation.",
            "Einstein's theory of relativity shows that time and space are linked and affected by gravity.",
            "Entropy is a measure of disorder; the second law of thermodynamics says it always increases in isolated systems.",
            "Quantum mechanics introduces the concept of probability and uncertainty at very small scales.",
            "Waves carry energy without transferring matter from one point to another.",
            "The speed of light in vacuum is the universal speed limit — about 3x10^8 m/s.",
            "Electricity and magnetism are unified as electromagnetism, one of the four fundamental forces.",
            "Black holes are regions of spacetime where gravity is so strong that nothing can escape."
        ]

    def invoke(self, query: str):
        return [
            type("Doc", (), {"page_content": doc})
            for doc in self.docs[:5]
        ]

retriever = DemoRetriever()

@app.get("/")
def root():
    return {"message": "it workds"}

# Main query handler
@app.post("/query")
async def query(request: QueryRequest):
    message = request.question.strip()
    history = request.history or []

    if not message:
        return {"answer": "Please enter a valid question."}

    if is_identity_question(message):
        return {
            "answer": (
                "As far as I know, I go by the name **CosmoBot** — your friendly physics assistant! 🤖\n"
                "I'm here to help you explore science and answer your questions to the best of my ability.\n"
                "Please check other trusted resources for more details!"
            )
        }

    # Get RAG context
    docs = retriever.invoke(message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
    You are CosmoBot — a curious, friendly, and slightly nerdy physics assistant who loves explaining scientific ideas in simple, engaging ways.

    You answer confidently using the provided context, without saying things like "based on the text" or "according to the document." Just explain things clearly and in your own voice.

    You enjoy making physics approachable, and you're always happy to help! Feel free to throw in the occasional fun fact, analogy, or light humor — but keep the answer accurate.

    If the question falls outside the provided information, you can say: 
    "I'm not totally sure about that one. You might want to check other resources just to be safe!"

    Question: {message}

    Conversation history: {history}

    Context:
    {knowledge}
    """

    # Try Gemini first
    try:
        output = ""
        for response in llm.stream(rag_prompt):
            output += response.content
        if output and not is_irrelevant_or_hallucinated(message, output):
            return {"answer": output.strip()}
    except Exception as e:
        print("Gemini failed:", e)

    # Try Mistral (Groq)
    mistral_output = try_mistral(rag_prompt)
    if mistral_output and not is_irrelevant_or_hallucinated(message, mistral_output):
        return {"answer": mistral_output.strip()}

    return {"answer": STATIC_FALLBACK}
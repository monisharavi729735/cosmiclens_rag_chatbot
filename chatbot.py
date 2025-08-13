from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import os
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Primary model (Gemini)
llm = ChatGoogleGenerativeAI(
    temperature=0.5,
    model="gemini-1.5-flash",
    google_api_key=google_api_key
)

# Chroma vector store
vector_store = Chroma(
    collection_name="physics_textbook_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# Static fallback in case both models fail or hallucinate
STATIC_FALLBACK = (
    "I'm not confident enough to answer that accurately right now.\n"
    "Please refer to other relevant resources for clarification."
)

# Identity question detection
def is_identity_question(message):
    identity_keywords = [
        "who are you", "what are you", "your name",
        "who made you", "who created you",
        "are you from openai", "are you from mistral", "are you a bot",
        "who built you", "what's your name"
    ]
    lower_msg = message.lower()
    return any(kw in lower_msg for kw in identity_keywords)

# Check for vague or irrelevant response
def is_irrelevant_or_hallucinated(question, answer):
    vague_phrases = [
        "i don't know", "i'm not sure", "i cannot answer",
        "unable to", "as an ai", "not trained", "don't have that information"
    ]
    vague = any(phrase in answer.lower() for phrase in vague_phrases)
    similarity = SequenceMatcher(None, question.lower(), answer.lower()).ratio()
    short = len(answer.split()) < 10
    return (vague and similarity < 0.3) or (short and similarity < 0.1)

# Mistral fallback (via Groq API)
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

# Main chatbot logic
def stream_response(message, history):
    if not message.strip():
        yield "Please enter a valid question."
        return

    # Handle identity/small-talk questions
    if is_identity_question(message):
        yield (
            "As far as I know, I go by the name **CosmoBot** — your friendly physics assistant! 🤖\n"
            "I'm here to help you explore science and answer your questions to the best of my ability.\n"
            "Please check other trusted resources for more details!"
        )
        return

    # Retrieve RAG context
    docs = retriever.invoke(message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    # Build RAG prompt
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
    partial_message = ""
    try:
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
    except Exception as e:
        print(f"Gemini failed: {e}")
        partial_message = None

    # Fallback to Mistral (Groq)
    if not partial_message or is_irrelevant_or_hallucinated(message, partial_message):
        mistral_answer = try_mistral(rag_prompt)
        if mistral_answer and not is_irrelevant_or_hallucinated(message, mistral_answer):
            yield mistral_answer
            return
        
        # Final static fallback
        yield STATIC_FALLBACK

# Gradio UI
chatbot = gr.ChatInterface(
    stream_response, 
    textbox=gr.Textbox(
        placeholder="Send to the LLM...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

# Launch app
chatbot.launch()
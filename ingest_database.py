from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

PDF_FILE_PATH = os.path.join("data", "relativity_notes.pdf")
CHROMA_PATH = "chroma_db"

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="physics_textbook_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

loader = PyPDFLoader(PDF_FILE_PATH)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(raw_documents)

uuids = [str(uuid4()) for _ in range(len(chunks))]

vector_store.add_documents(documents=chunks, ids=uuids)

print(f"Processed and stored {len(chunks)} chunks from the PDF.")

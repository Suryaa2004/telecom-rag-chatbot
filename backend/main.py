import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from pydantic import BaseModel


# fallback message
FALLBACK_MESSAGE = "I can only answer questions related to telecom plans and FAQs in my knowledge base."

# project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"

# persistent directory for Azure
CHROMA_DIR = Path("/home/chroma_db")

load_dotenv()

app = FastAPI()

# static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# vector store cache
vector_store = None


class ChatRequest(BaseModel):
    question: str


# -----------------------------
# Load documents
# -----------------------------
def load_documents():
    documents = []

    for file_path in DOCS_DIR.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    return documents


# -----------------------------
# Ingest documents into Chroma
# -----------------------------
def ingest_documents():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    store.persist()

    return store


# -----------------------------
# Get or create vector store
# -----------------------------
def get_vector_store():

    global vector_store

    if vector_store is not None:
        return vector_store

    # ensure directory exists
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # if embeddings already exist load them
    if any(CHROMA_DIR.iterdir()):
        vector_store = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=OpenAIEmbeddings(),
        )
    else:
        vector_store = ingest_documents()

    return vector_store


# -----------------------------
# Answer question
# -----------------------------
def answer_question(question: str) -> str:

    store = get_vector_store()

    results = store.similarity_search_with_score(question, k=3)

    if not results:
        return FALLBACK_MESSAGE

    best_score = results[0][1]

    if best_score > 1.2:
        return FALLBACK_MESSAGE

    context = "\n\n".join(doc.page_content for doc, _score in results).strip()

    if not context:
        return FALLBACK_MESSAGE

    prompt = f"""
You are a telecom support chatbot.

Answer ONLY using the provided context.

If the context does not contain the answer respond exactly with:
{FALLBACK_MESSAGE}

Context:
{context}

Question:
{question}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    return answer if answer else FALLBACK_MESSAGE


# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    get_vector_store()


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/chat")
async def chat(data: ChatRequest):
    return {"answer": answer_question(data.question)}
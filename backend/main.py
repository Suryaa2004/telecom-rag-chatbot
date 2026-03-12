from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.rag import SimpleRAGChatbot


BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
chatbot = None


class ChatRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/chat")
async def chat(data: ChatRequest):
    global chatbot
    if chatbot is None:
        chatbot = SimpleRAGChatbot()

    answer = chatbot.ask(data.question)
    return {"answer": answer}

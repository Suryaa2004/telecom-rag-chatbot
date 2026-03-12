# Telecom RAG Chatbot

This is a very simple Retrieval-Augmented Generation (RAG) chatbot project built with FastAPI, OpenAI, and ChromaDB. It only answers questions using the telecom plans and FAQs stored in the local knowledge base.

If the answer is not available in the knowledge base, the chatbot replies:

`I can only answer questions related to telecom plans and FAQs in my knowledge base.`

## Project Structure

```text
telecom-rag-chatbot/
├── docs/
│   ├── plans.txt
│   └── faq.txt
├── chroma_db/
├── backend/
│   ├── __init__.py
│   ├── ingest.py
│   ├── rag.py
│   └── main.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── .env
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment

```bash
python -m venv venv
```

2. Activate the virtual environment

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Add your OpenAI API key to `.env`

```env
OPENAI_API_KEY=your_openai_api_key_here
```

5. Run document ingestion

```bash
python backend/ingest.py
```

6. Start the FastAPI server

```bash
uvicorn backend.main:app --reload
```

7. Open the app in your browser

```text
http://127.0.0.1:8000
```

## How It Works

- `backend/ingest.py` loads text files from `docs/`, splits them into chunks, creates embeddings, and stores them in ChromaDB.
- `backend/rag.py` retrieves the most relevant chunks and sends the context plus the user question to the OpenAI model.
- `backend/main.py` provides the FastAPI routes.
- `templates/index.html` and `static/style.css` provide a minimal chat UI.

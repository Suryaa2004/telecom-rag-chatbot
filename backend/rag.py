import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI


FALLBACK_MESSAGE = "I can only answer questions related to telecom plans and FAQs in my knowledge base."
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"


class SimpleRAGChatbot:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=OpenAIEmbeddings(),
        )

    def ask(self, question: str) -> str:
        results = self.vector_store.similarity_search_with_score(question, k=3)
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
If the context does not contain the answer, respond exactly with:
{FALLBACK_MESSAGE}

Context:
{context}

Question:
{question}
""".strip()

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()
        return answer or FALLBACK_MESSAGE

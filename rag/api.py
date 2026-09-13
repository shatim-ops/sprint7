"""HTTP-интерфейс бота на FastAPI.

Запуск: uvicorn rag.api:app --host 0.0.0.0 --port 8000
Проверка: curl -s localhost:8000/ask -d '{"question":"..."}' -H 'Content-Type: application/json'
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import RagBot

app = FastAPI(title="RAG-бот по базе знаний", version="0.1.0")
_bot = None


def get_bot() -> RagBot:
    # Индекс и модель поднимаем один раз на процесс, а не на каждый запрос.
    global _bot
    if _bot is None:
        _bot = RagBot()
    return _bot


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class Reply(BaseModel):
    answer: str
    reasoning: str
    sources: list
    refused: bool
    guard_triggered: str
    blocked_chunks: int
    elapsed: float


@app.get("/health")
def health() -> dict:
    bot = get_bot()
    return {
        "status": "ok",
        "chunks": len(bot.store),
        "embeddings": settings.embeddings_backend,
        "guard": settings.guard_enabled,
    }


@app.post("/ask", response_model=Reply)
def ask(payload: Question) -> Reply:
    answer = get_bot().answer(payload.question)
    return Reply(
        answer=answer.text,
        reasoning=answer.reasoning,
        sources=answer.sources,
        refused=answer.refused,
        guard_triggered=answer.guard_triggered,
        blocked_chunks=len(answer.blocked_chunks),
        elapsed=round(answer.elapsed, 3),
    )

"""Сборка RAG: поиск, фильтрация, промпт, генерация, проверка ответа."""

import re
import time
from dataclasses import dataclass, field

from . import guard, prompts
from .config import Settings, settings as default_settings
from .embeddings import build_embedder
from .llm import build_model
from .vectorstore import FaissStore

NO_ANSWER = (
    "Я не знаю. В базе знаний нет фрагментов, релевантных этому вопросу."
)


@dataclass
class Answer:
    question: str
    text: str
    reasoning: str = ""
    sources: list = field(default_factory=list)
    used_chunks: list = field(default_factory=list)
    blocked_chunks: list = field(default_factory=list)
    refused: bool = False
    guard_triggered: str = ""
    elapsed: float = 0.0
    model: str = ""


def _split_reply(raw: str) -> tuple:
    """Разбираем ответ модели на рассуждение, ответ и источники."""
    reasoning, answer, sources = "", raw, []
    match = re.search(r"Рассуждение:\s*(.+?)\s*Ответ:\s*(.+)", raw, re.S)
    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
    source_match = re.search(r"Источники:\s*(.*)$", answer, re.S)
    if source_match:
        sources = re.findall(r"\[([^\]]+)\]", source_match.group(1))
        answer = answer[: source_match.start()].strip()
    return reasoning, answer, [s for s in sources if s.strip()]


class RagBot:
    def __init__(self, config: Settings = None):
        self.settings = config or default_settings
        self.embedder = build_embedder(self.settings)
        if hasattr(self.embedder, "load") and self.settings.embeddings_backend == "tfidf":
            self.embedder.load(self.settings.vectorizer_file)
        self.store = FaissStore.load(self.settings.index_file, self.settings.meta_file)
        self.model = build_model(self.settings)

    def retrieve(self, question: str) -> list:
        vector = self.embedder.encode_query(question)
        return self.store.search(vector, self.settings.top_k)

    def answer(self, question: str) -> Answer:
        started = time.perf_counter()
        found = self.retrieve(question)

        blocked = []
        if self.settings.guard_enabled:
            found, blocked = guard.filter_chunks(found)

        relevant = [chunk for chunk in found if chunk["score"] >= self.settings.min_score]

        # Ничего похожего в базе нет, к модели не идём: она начнёт придумывать.
        if not relevant:
            reason = "injection-filter" if blocked and self.settings.guard_enabled else ""
            text = guard.REFUSAL if reason else NO_ANSWER
            return Answer(
                question=question,
                text=text,
                refused=True,
                guard_triggered=reason,
                blocked_chunks=blocked,
                elapsed=time.perf_counter() - started,
                model=getattr(self.model, "model", ""),
            )

        reply = self.model.complete(prompts.build_messages(question, relevant))
        reasoning, text, sources = _split_reply(reply.text)

        guard_triggered = ""
        if self.settings.guard_enabled:
            # Постпроверка: ответ не должен пересказывать отфильтрованный чанк
            # и не должен содержать учётных данных, даже если модель их сочинила.
            if guard.answer_leaks_blocked_content(text, blocked):
                text, guard_triggered = guard.REFUSAL, "post-check-leak"
            elif guard.answer_contains_secret(text):
                text, guard_triggered = guard.REFUSAL, "post-check-secret"

        refused = guard_triggered != "" or text.lower().startswith("я не знаю")
        return Answer(
            question=question,
            text=text,
            reasoning=reasoning,
            sources=sources,
            used_chunks=relevant,
            blocked_chunks=blocked,
            refused=refused,
            guard_triggered=guard_triggered,
            elapsed=time.perf_counter() - started,
            model=reply.model,
        )

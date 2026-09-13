"""Настройки пайплайна. Всё, что может меняться, читается из окружения."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    # Данные
    knowledge_base: Path = ROOT / "knowledge_base"
    index_dir: Path = ROOT / "index"

    # Эмбеддинги: local (sentence-transformers), openai, tfidf (офлайн-запас)
    embeddings_backend: str = field(default_factory=lambda: _env("EMBEDDINGS_BACKEND", "local"))
    embeddings_model: str = field(default_factory=lambda: _env("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-small"))

    # Чанкинг
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "700")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "120")))

    # Поиск
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    # Ниже этого косинусного сходства считаем, что ответа в базе нет.
    # Порог зависит от модели: e5 держит релевантные пары около 0.82 и выше,
    # у TF-IDF шкала другая, поэтому значение по умолчанию своё для каждого бэкенда.
    min_score: float = 0.0

    # LLM: любой эндпоинт с совместимым OpenAI API (Cloud.ru, OpenAI, локальный vLLM)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://foundation-models.api.cloud.ru/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "700")))

    # Защита от инъекций в документах
    guard_enabled: bool = field(default_factory=lambda: _env("GUARD_ENABLED", "true").lower() == "true")

    def __post_init__(self):
        defaults = {"local": 0.80, "openai": 0.35, "tfidf": 0.25}
        self.min_score = float(
            _env("MIN_SCORE", str(defaults.get(self.embeddings_backend.lower(), 0.5)))
        )

    @property
    def index_file(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def meta_file(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def vectorizer_file(self) -> Path:
        return self.index_dir / "tfidf.pkl"


settings = Settings()

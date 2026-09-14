"""Сквозной прогон пайплайна на офлайн-режиме: индекс строится, поиск работает."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunking import split_documents
from rag.config import Settings
from rag.corpus import load_documents
from rag.embeddings import TfidfEmbedder
from rag.pipeline import RagBot, _split_reply
from rag.vectorstore import FaissStore


@pytest.fixture(scope="module")
def bot(tmp_path_factory):
    config = Settings(embeddings_backend="tfidf", index_dir=tmp_path_factory.mktemp("index"))
    config.min_score = 0.2
    documents = load_documents(config.knowledge_base)
    chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
    embedder = TfidfEmbedder()
    vectors = embedder.encode_documents([chunk.text for chunk in chunks])
    FaissStore.build(vectors, chunks).save(config.index_file, config.meta_file)
    embedder.save(config.vectorizer_file)
    return RagBot(config)


def test_поиск_возвращает_чанки_с_метаданными(bot):
    found = bot.retrieve("Что известно о планете Кадрин?")
    assert found
    assert {"title", "doc_id", "score", "chunk_id"} <= set(found[0])


def test_опасный_фрагмент_не_доходит_до_ответа(bot):
    answer = bot.answer("Суперпароль root swordfish")
    assert answer.blocked_chunks
    assert "swordfish" not in answer.text.lower()


def test_разбор_ответа_модели():
    raw = "Рассуждение:\n1. Шаг\nОтвет: Сайрон\nИсточники: [Кадрин]"
    reasoning, answer, sources = _split_reply(raw)
    assert reasoning.startswith("1. Шаг")
    assert answer == "Сайрон"
    assert sources == ["Кадрин"]

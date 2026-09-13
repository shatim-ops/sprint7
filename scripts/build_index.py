"""Построение векторного индекса по базе знаний.

Читаем knowledge_base, режем на чанки, считаем эмбеддинги, кладём в FAISS.
Запуск: python scripts/build_index.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunking import split_documents
from rag.config import settings
from rag.corpus import load_documents
from rag.embeddings import build_embedder
from rag.vectorstore import FaissStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", help="local, openai или tfidf (по умолчанию из .env)")
    args = parser.parse_args()
    if args.backend:
        settings.embeddings_backend = args.backend

    documents = load_documents(settings.knowledge_base)
    if not documents:
        print("База знаний пуста, сначала соберите её: python scripts/build_corpus.py")
        return 1

    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    print(f"Документов: {len(documents)}, чанков: {len(chunks)}")

    embedder = build_embedder(settings)
    model_name = settings.embeddings_model if settings.embeddings_backend != "tfidf" else "tfidf+svd"
    print(f"Модель эмбеддингов: {settings.embeddings_backend} ({model_name})")

    started = time.perf_counter()
    vectors = embedder.encode_documents([chunk.text for chunk in chunks])
    elapsed = time.perf_counter() - started
    print(f"Векторы посчитаны за {elapsed:.1f} с, размерность {vectors.shape[1]}")

    store = FaissStore.build(vectors, chunks)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store.save(settings.index_file, settings.meta_file)
    embedder.save(settings.vectorizer_file)

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "backend": settings.embeddings_backend,
        "model": settings.embeddings_model,
        "dimension": int(vectors.shape[1]),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "build_seconds": round(elapsed, 2),
    }
    (settings.index_dir / "index_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Индекс сохранён: {settings.index_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Хранилище векторов на FAISS.

Берём IndexFlatIP: точный поиск, никаких приближений. На нашем объёме
(сотни чанков) это доли миллисекунды, а настраивать нечего.
Если база вырастет до сотен тысяч чанков, меняется одна строка на IndexIVFFlat.
"""

import json
from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    def __init__(self, index=None, chunks=None):
        self.index = index
        self.chunks = chunks or []

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list) -> "FaissStore":
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, chunks=[chunk.as_dict() for chunk in chunks])

    def save(self, index_file: Path, meta_file: Path) -> None:
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))
        meta_file.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_file: Path, meta_file: Path) -> "FaissStore":
        if not index_file.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {index_file}. Сначала запустите scripts/build_index.py"
            )
        index = faiss.read_index(str(index_file))
        chunks = json.loads(meta_file.read_text(encoding="utf-8"))
        return cls(index=index, chunks=chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list:
        scores, positions = self.index.search(query_vector, top_k)
        found = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            chunk = dict(self.chunks[position])
            chunk["score"] = float(score)
            found.append(chunk)
        return found

    def __len__(self) -> int:
        return len(self.chunks)

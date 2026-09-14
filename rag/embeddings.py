"""Модели эмбеддингов.

Основной вариант по итогам исследования из Задания 1: локальная
multilingual-e5-small. Она даёт приемлемое качество на русском,
не требует отправки документов наружу и живёт на CPU.

Облачный вариант оставлен для сравнения, офлайновый TF-IDF нужен,
чтобы тесты и сборка работали там, где нет ни интернета, ни ключа.
"""

import pickle
from pathlib import Path

import numpy as np


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """FAISS ищет по скалярному произведению, поэтому нормируем векторы."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class LocalEmbedder:
    """sentence-transformers, модель скачивается один раз и работает локально."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # У семейства e5 префиксы обязательны, иначе качество проседает заметно.
        self.use_prefix = "e5" in model_name.lower()

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list) -> np.ndarray:
        prepared = [f"passage: {text}" for text in texts] if self.use_prefix else texts
        vectors = self.model.encode(prepared, batch_size=32, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype="float32"))

    def encode_query(self, text: str) -> np.ndarray:
        prepared = f"query: {text}" if self.use_prefix else text
        vector = self.model.encode([prepared], show_progress_bar=False)
        return _normalize(np.asarray(vector, dtype="float32"))

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> None:
        return None


class OpenAIEmbedder:
    """Облачные эмбеддинги через совместимый с OpenAI эндпоинт."""

    def __init__(self, model_name: str, base_url: str, api_key: str):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._dimension = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: list) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        vectors = np.asarray([item.embedding for item in response.data], dtype="float32")
        self._dimension = vectors.shape[1]
        return _normalize(vectors)

    def encode_documents(self, texts: list) -> np.ndarray:
        # Батчим, потому что у провайдеров есть лимит на размер запроса.
        parts = [self._encode(texts[i:i + 64]) for i in range(0, len(texts), 64)]
        return np.vstack(parts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text])

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> None:
        return None


class TfidfEmbedder:
    """Запасной вариант без сети: TF-IDF по символьным n-граммам плюс SVD.

    Качество ниже, чем у нейросетевых эмбеддингов, зато не требует
    ни загрузки модели, ни ключа. Используется в тестах и в CI.
    """

    def __init__(self, dimension: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.svd = TruncatedSVD(n_components=dimension, random_state=17)
        self._dimension = dimension
        self.fitted = False

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: list) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._dimension = min(self._dimension, matrix.shape[1] - 1, len(texts) - 1)
        self.svd.n_components = self._dimension
        reduced = self.svd.fit_transform(matrix)
        self.fitted = True
        return _normalize(np.asarray(reduced, dtype="float32"))

    def encode_query(self, text: str) -> np.ndarray:
        matrix = self.vectorizer.transform([text])
        reduced = self.svd.transform(matrix)
        return _normalize(np.asarray(reduced, dtype="float32"))

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps({"vectorizer": self.vectorizer, "svd": self.svd}))

    def load(self, path: Path) -> None:
        state = pickle.loads(path.read_bytes())
        self.vectorizer = state["vectorizer"]
        self.svd = state["svd"]
        self._dimension = self.svd.n_components
        self.fitted = True


def build_embedder(settings):
    backend = settings.embeddings_backend.lower()
    if backend == "local":
        return LocalEmbedder(settings.embeddings_model)
    if backend == "openai":
        return OpenAIEmbedder(settings.embeddings_model, settings.llm_base_url, settings.llm_api_key)
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Неизвестный backend эмбеддингов: {settings.embeddings_backend}")

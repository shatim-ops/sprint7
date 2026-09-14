"""Чтение базы знаний с диска."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    category: str
    path: str
    text: str


def _parse_front_matter(raw: str) -> tuple:
    if not raw.startswith("---"):
        return {}, raw
    _, block, body = raw.split("---", 2)
    meta = {}
    for line in block.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def load_documents(directory: Path) -> list:
    """Забираем все markdown и txt из папки базы знаний.

    Подпапки тоже читаем: в knowledge_base/injected лежит документ
    с инъекцией, он должен попадать в индекс наравне с остальными.
    """
    documents = []
    paths = sorted(list(directory.rglob("*.md")) + list(directory.rglob("*.txt")))
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        documents.append(
            Document(
                doc_id=meta.get("doc_id", path.stem),
                title=meta.get("title", path.stem),
                category=meta.get("category", "Прочее"),
                path=str(path.relative_to(directory.parent)),
                text=body,
            )
        )
    return documents

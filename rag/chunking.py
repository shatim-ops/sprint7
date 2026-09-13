"""Разбиение документов на чанки.

Режем рекурсивным сплиттером LangChain по абзацам, а не по символам наугад:
так внутри чанка остаётся законченная мысль, и цитата в ответе читается.
Каждому чанку сохраняем источник и порядковый номер, чтобы бот мог сослаться.
"""

from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .corpus import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    path: str
    position: int
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


def split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
        keep_separator=True,
    )
    chunks = []
    for document in documents:
        for position, piece in enumerate(splitter.split_text(document.text)):
            piece = piece.strip()
            if len(piece) < 40:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#{position}",
                    doc_id=document.doc_id,
                    title=document.title,
                    category=document.category,
                    path=document.path,
                    position=position,
                    text=piece,
                )
            )
    return chunks

# RAG-бот по корпоративной базе знаний

Проектная работа 7 спринта курса "Архитектор программного обеспечения".
Кейс компании QuantumForge Software: бот отвечает на вопросы сотрудников
по внутренней базе знаний и честно говорит "не знаю", когда данных нет.

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # вписать LLM_API_KEY
python scripts/build_corpus.py   # собрать базу знаний из исходников
python scripts/build_index.py    # построить FAISS-индекс
python -m rag.cli                # консольный бот
```

Ответы на задания собраны в [Project_template.md](Project_template.md).

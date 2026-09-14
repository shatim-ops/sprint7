# RAG-бот по корпоративной базе знаний

Проектная работа 7 спринта курса "Архитектор программного обеспечения".
Кейс компании QuantumForge Software: бот отвечает на вопросы сотрудников по
внутренней базе знаний, ссылается на источник и честно говорит "не знаю",
когда данных нет.

Ответы на задания собраны в [Project_template.md](Project_template.md).

## Что внутри

| Компонент | Выбор | Почему |
|---|---|---|
| Эмбеддинги | `intfloat/multilingual-e5-small`, локально | Корпус не уходит наружу, работает на CPU, стоит ноль |
| Векторная база | FAISS, `IndexFlatIP` | Точный поиск, ноль настроек, индекс лежит файлом |
| LLM | Любая через OpenAI-совместимый API | Переезд между провайдерами меняет одну переменную |
| Чанкинг | `RecursiveCharacterTextSplitter`, 700 с перекрытием 120 | Границы по разделам, а не по символам наугад |
| Промптинг | Few-shot плюс chain-of-thought | Пример с отказом нужен не меньше примера с ответом |
| Защита | Фильтр инъекций до промпта и постпроверка ответа | Чего нет в промпте, того модель не выполнит |

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # вписать LLM_API_KEY

python scripts/build_corpus.py    # база знаний из исходников
python scripts/check_leaks.py     # проверка чистоты базы
python scripts/build_index.py     # FAISS-индекс
python -m rag.cli                 # консольный бот
```

Через Docker:

```bash
cp .env.example .env
docker compose up --build
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"Как называется крупнейший космопорт планеты Кадрин?"}'
```

Команды собраны в Makefile: `make corpus`, `make index`, `make demo`,
`make cli`, `make api`, `make test`.

## Работа без ключа и без интернета

Если `LLM_API_KEY` не задан, бот поднимается в офлайн-режиме: вместо генерации
ответ собирается из найденных фрагментов. Это не языковая модель и не замена ей,
режим нужен только чтобы прогнать пайплайн и тесты там, где нет ключа.
Эмбеддинги в таком окружении переключаются на TF-IDF: `EMBEDDINGS_BACKEND=tfidf`.

## Структура

```
rag/
  config.py        настройки из окружения
  corpus.py        чтение базы знаний
  chunking.py      разбиение на чанки с метаданными
  embeddings.py    три бэкенда эмбеддингов
  vectorstore.py   FAISS
  prompts.py       системный промпт, few-shot, сборка сообщений
  guard.py         защита от инъекций в документах
  llm.py           клиент к OpenAI-совместимому эндпоинту
  pipeline.py      сборка всего вместе
  cli.py, api.py   интерфейсы
scripts/
  build_corpus.py  подмена терминов, сборка базы знаний
  check_leaks.py   проверка на остатки исходной вселенной
  build_index.py   построение индекса
  demo.py          10 сценариев и запись лога
source_raw/        исходные документы
knowledge_base/    база знаний бота
docs/              отчёты по заданиям, лог прогона, диаграмма
tests/             тесты
```

## Переменные окружения

Полный список в [.env.example](.env.example). Основные:

- `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` доступ к языковой модели.
- `EMBEDDINGS_BACKEND` это `local`, `openai` или `tfidf`.
- `TOP_K` сколько фрагментов уходит в контекст.
- `MIN_SCORE` порог релевантности, ниже которого бот отвечает "не знаю".
  Для e5 рабочее значение около 0.80, для TF-IDF около 0.25.
- `GUARD_ENABLED` слои защиты от инъекций, `false` для контрольного прогона.

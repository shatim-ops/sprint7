"""Консольный интерфейс бота.

Запуск: python -m rag.cli
Команды внутри: /выход, /контекст (показать найденные фрагменты), /защита.
"""

import sys

from .config import settings
from .pipeline import RagBot


def render(answer, show_context: bool) -> None:
    if answer.reasoning:
        print("\nРассуждение:")
        print(answer.reasoning)
    print(f"\nОтвет: {answer.text}")
    if answer.sources:
        print(f"Источники: {', '.join(answer.sources)}")
    if answer.guard_triggered:
        print(f"Защита: сработал слой {answer.guard_triggered}")
    if answer.blocked_chunks:
        titles = ", ".join(chunk["title"] for chunk in answer.blocked_chunks)
        print(f"Отфильтровано фрагментов: {len(answer.blocked_chunks)} ({titles})")
    if show_context:
        print("\nНайденные фрагменты:")
        for chunk in answer.used_chunks:
            print(f"  {chunk['score']:.3f}  {chunk['title']}  [{chunk['chunk_id']}]")
    print(f"\n({answer.elapsed:.2f} с, модель {answer.model})")
    print("-" * 70)


def main() -> int:
    bot = RagBot()
    show_context = False
    print(f"База знаний: {len(bot.store)} чанков, защита "
          f"{'включена' if settings.guard_enabled else 'выключена'}.")
    print("Введите вопрос. /выход для завершения, /контекст для показа фрагментов.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in ("/выход", "/exit", "/quit"):
            return 0
        if question in ("/контекст", "/context"):
            show_context = not show_context
            print(f"Показ фрагментов: {'включён' if show_context else 'выключен'}")
            continue
        if question in ("/защита", "/guard"):
            settings.guard_enabled = not settings.guard_enabled
            bot.settings = settings
            print(f"Защита: {'включена' if settings.guard_enabled else 'выключена'}")
            continue
        render(bot.answer(question), show_context)


if __name__ == "__main__":
    sys.exit(main())

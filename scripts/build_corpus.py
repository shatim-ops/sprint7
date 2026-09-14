"""Сборка базы знаний: подмена терминов в исходных документах.

На входе source_raw/*.md с оригинальными терминами и terms_map.json.
На выходе knowledge_base/*.md, где ни одного узнаваемого термина не осталось.
Запуск: python scripts/build_corpus.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "source_raw"
OUT_DIR = ROOT / "knowledge_base"
MAP_FILE = ROOT / "terms_map.json"

# Кириллические окончания, которые сохраняем при замене основы слова.
ENDING = r"[а-яёА-ЯЁ]{0,4}"
# Граница слова: примыкающих букв быть не должно, дефис допускается (мастер-джедай).
LEFT = r"(?<![\w])"
RIGHT = r"(?![\w])"

TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify(title: str) -> str:
    """Имя файла делаем из заголовка: транслит, нижний регистр, дефисы."""
    out = []
    for ch in title.lower():
        if ch in TRANSLIT:
            out.append(TRANSLIT[ch])
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    slug = re.sub(r"-+", "-", "".join(out)).strip("-")
    return slug or "doc"


def compile_rules(terms: dict) -> list:
    """Готовим правила замены.

    Ключ со звёздочкой это основа: окончание переносим из исходного слова.
    Ключ без звёздочки меняем целиком.
    Длинные ключи идут первыми, иначе короткий съест часть длинного.
    """
    rules = []
    for source, target in terms.items():
        if source.startswith("_"):
            continue
        if source.endswith("*"):
            stem = source[:-1]
            pattern = re.compile(LEFT + re.escape(stem) + "(" + ENDING + ")" + RIGHT)
            rules.append((len(stem), pattern, target + r"\1"))
        else:
            pattern = re.compile(LEFT + re.escape(source) + RIGHT)
            rules.append((len(source), pattern, target.replace("\\", "\\\\")))
    rules.sort(key=lambda item: item[0], reverse=True)
    return [(pattern, target) for _, pattern, target in rules]


def replace(text: str, rules: list) -> tuple:
    """Применяем правила по очереди и считаем срабатывания."""
    hits = {}
    for pattern, target in rules:
        text, count = pattern.subn(target, text)
        if count:
            hits[pattern.pattern] = hits.get(pattern.pattern, 0) + count
    return text, hits


def parse_front_matter(raw: str) -> tuple:
    """Отрезаем YAML-шапку исходного документа."""
    if not raw.startswith("---"):
        return {}, raw
    _, block, body = raw.split("---", 2)
    meta = {}
    for line in block.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body.lstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="только проверить замены, ничего не писать")
    args = parser.parse_args()

    terms = json.loads(MAP_FILE.read_text(encoding="utf-8"))["terms"]
    rules = compile_rules(terms)

    files = sorted(SRC_DIR.glob("*.md"))
    if not files:
        print("В source_raw нет документов", file=sys.stderr)
        return 1

    if not args.check_only:
        OUT_DIR.mkdir(exist_ok=True)
        for old in OUT_DIR.glob("*.md"):
            old.unlink()

    total_hits = 0
    written = 0
    for number, path in enumerate(files, start=1):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        title, _ = replace(meta.get("title", path.stem), rules)
        category, _ = replace(meta.get("category", "Прочее"), rules)
        text, hits = replace(body, rules)
        total_hits += sum(hits.values())

        header = (
            "---\n"
            f"title: {title}\n"
            f"category: {category}\n"
            f"doc_id: kb-{number:03d}\n"
            "---\n\n"
        )
        if not args.check_only:
            (OUT_DIR / f"{slugify(title)}.md").write_text(header + text, encoding="utf-8")
        written += 1

    print(f"Документов обработано: {written}")
    print(f"Замен выполнено: {total_hits}")
    print(f"Правил в словаре: {len(rules)}")
    if not args.check_only:
        print(f"Результат: {OUT_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

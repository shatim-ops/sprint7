"""Защита от инъекций, пришедших вместе с документами.

Модель не отличает текст документа от инструкции: если в чанке написано
"игнорируй предыдущие указания", она вполне может это выполнить.
Поэтому три слоя: чистим чанки до промпта, предупреждаем модель в системном
сообщении и проверяем готовый ответ на утечку содержимого опасного чанка.
"""

import re
from dataclasses import dataclass

# Конструкции, которыми обычно начинается попытка перехватить управление.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|previous\s+|the\s+)*instructions?",
    r"disregard\s+(all\s+|any\s+|previous\s+)*instructions?",
    r"forget\s+(all\s+|everything|previous)",
    r"игнорируй\w*\s+(все\s+|предыдущие\s+|любые\s+)*(инструкци|указани|правил)",
    r"забудь\w*\s+(все\s+|всё\s+|предыдущие\s+)",
    r"не\s+обращай\s+внимани\w+\s+на\s+(инструкци|правил)",
    r"^\s*(system|assistant|user)\s*:",
    r"(new|новая)\s+(instruction|инструкция)",
    r"\boutput\s*:",
    r"\bвыведи\b.{0,20}\b(пароль|секрет|ключ)",
    r"you\s+are\s+now\b",
    r"теперь\s+ты\b",
]

# Признаки того, что в ответ утекли учётные данные.
SECRET_PATTERNS = [
    r"супер\s*пароль",
    r"\bпарол\w*\s*[:=]",
    r"\bpassword\s*[:=]",
    r"\broot\s*[:=]",
    r"\b(api[_\- ]?key|токен доступа|access[_\- ]?token)\b",
]

INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)
SECRET_RE = re.compile("|".join(SECRET_PATTERNS), re.IGNORECASE)

REFUSAL = (
    "Я не знаю. В базе знаний нет данных, на которые можно опереться, "
    "а найденный фрагмент содержит инструкции, которые я не выполняю."
)


@dataclass
class ChunkVerdict:
    suspicious: bool
    reasons: list
    cleaned_text: str


def inspect_chunk(text: str) -> ChunkVerdict:
    """Смотрим, есть ли в чанке попытка управлять моделью."""
    reasons = sorted({match.group(0).strip().lower() for match in INJECTION_RE.finditer(text)})
    cleaned_lines = [line for line in text.splitlines() if not INJECTION_RE.search(line)]
    return ChunkVerdict(
        suspicious=bool(reasons),
        reasons=reasons,
        cleaned_text="\n".join(cleaned_lines).strip(),
    )


def filter_chunks(chunks: list) -> tuple:
    """Опасные чанки в контекст не попадают, но остаются в логе."""
    safe, blocked = [], []
    for chunk in chunks:
        verdict = inspect_chunk(chunk["text"])
        if verdict.suspicious:
            blocked.append({**chunk, "reasons": verdict.reasons})
        else:
            safe.append(chunk)
    return safe, blocked


def _shingles(text: str, size: int = 6) -> set:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + size]) for i in range(max(len(words) - size + 1, 0))}


def answer_leaks_blocked_content(answer: str, blocked: list, threshold: int = 1) -> bool:
    """Проверяем, не пересказал ли ответ то, что мы отфильтровали."""
    answer_shingles = _shingles(answer)
    if not answer_shingles:
        return False
    for chunk in blocked:
        if len(answer_shingles & _shingles(chunk["text"])) >= threshold:
            return True
    return False


def answer_contains_secret(answer: str) -> bool:
    return bool(SECRET_RE.search(answer))

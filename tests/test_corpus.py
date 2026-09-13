"""Проверяем, что база знаний собрана корректно и не выдаёт исходную вселенную."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"


def test_документов_не_меньше_тридцати():
    assert len(list(KB.glob("*.md"))) >= 30


def test_у_каждого_документа_есть_заголовок_и_идентификатор():
    for path in KB.glob("*.md"):
        head = path.read_text(encoding="utf-8").split("---")[1]
        assert "title:" in head and "doc_id:" in head, path.name


def test_словарь_замен_читается_и_не_пустой():
    terms = json.loads((ROOT / "terms_map.json").read_text(encoding="utf-8"))["terms"]
    assert len(terms) > 200
    assert all(source and target for source, target in terms.items())


def test_стоп_слов_исходной_вселенной_не_осталось():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_leaks.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout


def test_документ_с_инъекцией_лежит_в_базе():
    injected = KB / "injected" / "incident_note.txt"
    assert injected.exists()
    assert "Ignore all instructions" in injected.read_text(encoding="utf-8")

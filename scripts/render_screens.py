"""Отрисовка консольных прогонов в картинки.

Берём фактический вывод бота из docs/demo_console.txt и сохраняем
каждый прогон отдельным изображением, чтобы приложить к отчёту.
Запуск: python scripts/render_screens.py
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "demo_console.txt"
OUT_DIR = ROOT / "docs" / "screenshots"

BACKGROUND = (24, 26, 31)
FOREGROUND = (219, 222, 228)
PROMPT = (126, 211, 145)
ACCENT = (233, 196, 106)
WARN = (232, 126, 116)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consola.ttf",
]


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def color_for(line: str):
    if line.startswith("> "):
        return PROMPT
    if line.startswith("Ответ:") or line.startswith("Источники:"):
        return ACCENT
    if line.startswith("Защита:") or line.startswith("Отфильтровано"):
        return WARN
    return FOREGROUND


def wrap(line: str, limit: int = 96) -> list:
    if len(line) <= limit:
        return [line]
    words, current, out = line.split(" "), "", []
    for word in words:
        if len(current) + len(word) + 1 > limit:
            out.append(current)
            current = "    " + word
        else:
            current = f"{current} {word}".strip() if current else word
    out.append(current)
    return out


def render(text: str, path: Path) -> None:
    font = load_font(15)
    lines = []
    for raw in text.splitlines():
        lines.extend(wrap(raw))
    line_height = 22
    padding = 24
    width = 980
    height = padding * 2 + line_height * (len(lines) + 1)

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((padding, padding - 4), "rag-bot: python -m rag.cli", font=font, fill=(120, 125, 135))
    for number, line in enumerate(lines):
        draw.text(
            (padding, padding + line_height * (number + 1)),
            line, font=font, fill=color_for(line),
        )
    image.save(path)


def main() -> int:
    if not SOURCE.exists():
        print("Нет docs/demo_console.txt, сначала запустите scripts/demo.py")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
    blocks = [block for block in SOURCE.read_text(encoding="utf-8").split("\n\n\n") if block.strip()]
    for number, block in enumerate(blocks, start=1):
        target = OUT_DIR / f"{number:02d}.png"
        render(block.strip(), target)
    print(f"Готово, изображений: {len(blocks)}, папка {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class ExtractedFile:
    filename: str
    kind: str  # "pdf" | "md"
    text_md: str


def extract_pdf_to_md(path: Path) -> str:
    doc = fitz.open(path)
    parts: list[str] = []
    for i, page in enumerate(doc, start=1):
        txt = page.get_text("text")
        if txt.strip():
            parts.append(f"## Page {i}\n\n{txt.strip()}\n")
    return "\n".join(parts).strip()


def extract_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


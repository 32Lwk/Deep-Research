from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def pandoc_available(pandoc_path: str = "pandoc") -> bool:
    return shutil.which(pandoc_path) is not None


def md_to_pdf(
    md_path: Path,
    pdf_path: Path,
    *,
    pandoc_path: str = "pandoc",
    pdf_engine: str | None = None,
    mainfont: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> None:
    if not pandoc_available(pandoc_path):
        raise FileNotFoundError(f"pandoc not found: {pandoc_path}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [pandoc_path, str(md_path), "-o", str(pdf_path)]
    if pdf_engine:
        cmd += ["--pdf-engine", pdf_engine]
    if mainfont:
        cmd += ["-V", f"mainfont={mainfont}"]
    if extra_args:
        cmd += list(extra_args)
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )


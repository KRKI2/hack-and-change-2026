"""Собирает dist/submission.zip для загрузки на платформу.

В архив попадают solution.py, пакет legal_rag, кэш эмбеддингов (если рассчитан)
и минимальный requirements.txt. Секреты и .env не включаются.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "submission.zip"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "solution.py", "solution.py")
        for file in sorted((ROOT / "legal_rag").glob("*.py")):
            z.write(file, f"legal_rag/{file.name}")
        cache = ROOT / "cache" / "embeddings.json"
        if cache.is_file():
            z.write(cache, "cache/embeddings.json")
        z.writestr("requirements.txt", "httpx==0.28.1\n")
    names = zipfile.ZipFile(OUT).namelist()
    print(f"Создан {OUT} ({OUT.stat().st_size // 1024} КБ): {', '.join(names)}")


if __name__ == "__main__":
    main()

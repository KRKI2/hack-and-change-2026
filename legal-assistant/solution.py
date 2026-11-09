"""Точка входа для автоматической проверки на платформе хакатона.

Читает data/questions.csv (id,question) и базу знаний data/kb.zip,
пишет submission.csv (id,answer; все поля в кавычках).
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from legal_rag import config as cfg  # noqa: E402
from legal_rag import llm  # noqa: E402
from legal_rag.assistant import FALLBACK  # noqa: E402
from legal_rag.runtime import build_assistant  # noqa: E402


def find_questions() -> Path:
    here = Path(__file__).resolve().parent
    for root in (cfg.DATA_DIR, here / cfg.DATA_DIR, Path.cwd(), here):
        path = root / "questions.csv"
        if path.is_file():
            return path
    raise FileNotFoundError("Не найден questions.csv")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
        return [(str(r["id"]).strip(), str(r["question"] or "").strip()) for r in reader]


def main() -> None:
    if not cfg.API_BASE or not cfg.LLM_KEY:
        raise ValueError("Нужны LLM_BASE_URL и LLM_API_KEY (или их аналоги)")
    questions = load_questions(find_questions())
    assistant = build_assistant()
    print(f"Вопросов: {len(questions)}", flush=True)

    def work(i: int, question: str) -> tuple[int, str]:
        try:
            with llm.make_client() as client:
                return i, assistant.answer(client, question).text
        except Exception as exc:  # noqa: BLE001 — один вопрос не должен ронять прогон
            print(f"Ошибка вопроса {i}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return i, FALLBACK

    answers = [FALLBACK] * len(questions)
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
        futures = [pool.submit(work, i, q) for i, (_, q) in enumerate(questions)]
        for done, future in enumerate(as_completed(futures), 1):
            i, text = future.result()
            answers[i] = " ".join(text.split())  # одна физическая строка CSV
            if done % 10 == 0 or done == len(questions):
                print(f"Обработано {done}/{len(questions)}", flush=True)

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for (qid, _), text in zip(questions, answers):
            writer.writerow([qid, text])
    print(f"Создан submission.csv: {len(questions)} ответов", flush=True)


if __name__ == "__main__":
    main()

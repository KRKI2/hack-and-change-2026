"""Локальная проверка на демо-вопросах: ответы бота рядом с эталоном + оценка LLM-судьёй.

Запуск:  python tools/evaluate.py [questions.csv] [reference.csv]
Результат: eval_report.md
"""

from __future__ import annotations

import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legal_rag import config as cfg  # noqa: E402
from legal_rag import llm  # noqa: E402
from legal_rag.runtime import build_assistant  # noqa: E402

JUDGE = """Ты — судья. Сравни ответ ассистента с эталонным ответом на юридический вопрос.
Оцени от 0 до 10: фактическая точность (цифры, сроки, пункты), полнота относительно эталона,
корректность отказа/исправления ошибочной предпосылки, отсутствие выдуманных фактов.
Ответь строго в формате: SCORE: <число>\nREASON: <одно предложение>"""


def read(path: Path, key: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r[key] for r in csv.DictReader(f)}


def main() -> None:
    qpath = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "questions.csv"
    rpath = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "demo_output.csv"
    questions = read(qpath, "question")
    reference = read(rpath, "answer") if rpath.is_file() else {}
    assistant = build_assistant()

    def run(item: tuple[str, str]) -> tuple[str, str, float, int, str]:
        qid, q = item
        with llm.make_client() as client:
            t = time.time()
            ans = assistant.answer(client, q).text
            took = time.time() - t
            score, reason = -1, ""
            if qid in reference:
                try:
                    verdict = llm.chat(client, [{"role": "system", "content": JUDGE}, {"role": "user", "content":
                        f"ВОПРОС:\n{q}\n\nЭТАЛОН:\n{reference[qid]}\n\nОТВЕТ АССИСТЕНТА:\n{ans}"}], max_tokens=200)
                    m = re.search(r"SCORE:\s*(\d+)", verdict)
                    score = int(m.group(1)) if m else -1
                    reason = (re.search(r"REASON:\s*(.*)", verdict) or [None, ""])[1]
                except llm.ApiError as exc:
                    reason = f"судья недоступен: {exc}"
        print(f"[{qid}] {took:.1f} с, оценка {score}", flush=True)
        return qid, ans, took, score, reason

    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
        results = list(pool.map(run, questions.items()))

    scored = [r[3] for r in results if r[3] >= 0]
    lines = [f"# Отчёт локальной проверки\n",
             f"Вопросов: {len(results)}; средняя оценка судьи: {sum(scored) / max(1, len(scored)):.2f}/10; "
             f"среднее время: {sum(r[2] for r in results) / max(1, len(results)):.1f} с\n"]
    for qid, ans, took, score, reason in results:
        lines += [f"## {qid} — оценка {score}/10, {took:.1f} с", f"**Вопрос:** {questions[qid]}", "",
                  f"**Ответ:** {ans}", "", f"**Эталон:** {reference.get(qid, '—')}", "", f"_Судья: {reason}_", ""]
    (ROOT / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Средняя оценка: {sum(scored) / max(1, len(scored)):.2f}/10 → eval_report.md")


if __name__ == "__main__":
    main()

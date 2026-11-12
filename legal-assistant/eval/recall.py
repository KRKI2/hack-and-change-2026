"""Recall@k поиска по разметке gold_docs из eval/questions.csv (для роли «данные и поиск»).

В отличие от run_eval.py этот скрипт импортирует код поиска, поэтому после
перестройки модулей нужно поправить ОДНУ функцию — build_retriever().
Без ключа API считает только поиск по словам (BM25); с ключом — гибридный.

    python eval/recall.py                 # таблица по категориям + промахи
    python eval/recall.py --misses 30     # показать больше промахов
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
sys.path.insert(0, str(ROOT))
KS = (1, 3, 5)


def build_retriever():
    """Адаптер к текущему коду поиска. Возвращает функцию:
    question -> (документы по убыванию релевантности, документы в итоговом контексте, размер контекста в символах).
    Документы — имена файлов базы."""
    from legal_rag import config as cfg
    from legal_rag import llm
    from legal_rag.assistant import parse_dialog
    from legal_rag.kb import find_kb_path, load_kb
    from legal_rag.search import Searcher

    kb = load_kb(find_kb_path(ROOT / "data", cfg.KB_DIR))
    online = bool(cfg.API_BASE and cfg.EMBED_KEY)
    client = llm.make_client() if online else None
    searcher = Searcher(kb, client)
    mode = "гибридный (BM25 + эмбеддинги)" if searcher.vectors else "только BM25"

    def retrieve(question: str) -> tuple[list[str], list[str], int]:
        turns = parse_dialog(question)
        focus = turns[-1].content
        query = " ".join(t.content for t in turns)
        result = searcher.retrieve(client, query, focus)
        ranked: list[str] = []
        for h in result.hits:
            name = kb.by_id[h.chunk.doc_id].name
            if name not in ranked:
                ranked.append(name)
        return ranked, [d.name for d, _ in result.documents], len(result.context)

    return retrieve, mode


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        value = value.strip().strip('"').strip("'")
        if sep and value and not key.startswith("#") and not os.environ.get(key.strip()):
            os.environ[key.strip()] = value


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Recall@k поиска по gold_docs.")
    ap.add_argument("--questions", type=Path, default=EVAL / "questions.csv")
    ap.add_argument("--misses", type=int, default=15, help="сколько промахов показать")
    args = ap.parse_args()

    load_env_file(ROOT / ".env")
    with args.questions.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("gold_docs", "").strip()]
    retrieve, mode = build_retriever()

    stats = defaultdict(lambda: defaultdict(float))
    misses = []
    for r in rows:
        gold = [g.strip() for g in r["gold_docs"].split(";") if g.strip()]
        ranked, context, chars = retrieve(r["question"])
        found = lambda names: sum(any(g in n for n in names) for g in gold) / len(gold)  # noqa: E731
        for scope in (r["category"], "ВСЕГО"):
            s = stats[scope]
            s["n"] += 1
            for k in KS:
                s[f"r{k}"] += found(ranked[:k])
            s["ctx"] += found(context)
            s["docs"] += len(context)
            s["chars"] += chars
        if found(context) < 1:
            lost = [g for g in gold if not any(g in n for n in context)]
            misses.append(f"{r['id']}: нет в контексте — {'; '.join(lost)} | {r['question'][:90].replace(chr(10), ' ')}")

    print(f"Режим поиска: {mode}. Вопросов с разметкой: {len(rows)}\n")
    print(f"{'категория':14} {'n':>3} " + " ".join(f"{'R@' + str(k):>6}" for k in KS)
          + f" {'в контексте':>12} {'док-тов':>8} {'символов':>9}")
    for scope, s in stats.items():
        n = s["n"]
        print(f"{scope:14} {int(n):>3} " + " ".join(f"{100 * s[f'r{k}'] / n:>5.0f}%" for k in KS)
              + f" {100 * s['ctx'] / n:>11.0f}% {s['docs'] / n:>8.1f} {s['chars'] / n:>9.0f}")
    print("\nR@k — доля нужных документов среди первых k по ранжированию фрагментов;")
    print("«в контексте» — доля нужных документов в итоговом контексте для модели;")
    print("«док-тов» и «символов» — средний размер контекста (чем меньше при том же recall, тем лучше).")
    if misses:
        print(f"\nПромахи ({len(misses)}):")
        for m in misses[: args.misses]:
            print("  " + m)


if __name__ == "__main__":
    main()

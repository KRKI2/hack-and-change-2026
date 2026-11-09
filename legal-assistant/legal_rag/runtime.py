"""Сборка ассистента: загрузка базы, индекс, клиент API."""

from __future__ import annotations

from . import config as cfg
from . import llm
from .assistant import LegalAssistant
from .kb import find_kb_path, load_kb
from .search import Searcher


def build_assistant() -> LegalAssistant:
    kb = load_kb(find_kb_path(cfg.DATA_DIR, cfg.KB_DIR))
    with llm.make_client() as client:
        searcher = Searcher(kb, client if cfg.API_BASE and cfg.EMBED_KEY else None)
    print(f"[runtime] документов: {len(kb.documents)}, фрагментов: {len(kb.chunks)}, "
          f"эмбеддинги: {'да' if searcher.vectors else 'нет'}", flush=True)
    return LegalAssistant(kb, searcher)

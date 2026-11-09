"""Настройки из переменных окружения (имена совместимы с платформой хакатона)."""

from __future__ import annotations

import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float = 0.0, high: float = 2.0) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = env(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on", "да"}


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(env("DATA_DIR", "data"))
KB_DIR = env("KB_DIR")  # папка с документами (альтернатива kb.zip)
CACHE_DIR = Path(env("CACHE_DIR", str(ROOT / "cache")))

API_BASE = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
EMBED_BASE = (env("EMBEDDER_BASE_URL") or API_BASE).rstrip("/")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_BASE = (env("RERANKER_BASE_URL") or API_BASE).rstrip("/")
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY

LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "cohere/rerank-5-fast")

USE_EMBEDDINGS = env_bool("USE_EMBEDDINGS", True)
USE_RERANKER = env_bool("USE_RERANKER", True)
VERIFY = env_bool("VERIFY", True)  # отдельный этап самопроверки ответа

TOP_CHUNKS = env_int("TOP_CHUNKS", 10, 3, 40)
RERANKER_TOP_N = env_int("RERANKER_TOP_N", 8, 1, 30)
SMALL_DOC_CHARS = env_int("SMALL_DOC_CHARS", 9000, 0, 200000)  # небольшие документы даются целиком
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 45000, 5000, 400000)

MAX_TOKENS = env_int("MAX_TOKENS", 1500, 128, 8192)
MAX_WORKERS = env_int("MAX_WORKERS", 4, 1, 16)
TEMPERATURE = env_float("TEMPERATURE", 0.0)
HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 120.0, 5.0, 600.0)

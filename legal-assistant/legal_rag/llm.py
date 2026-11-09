"""Клиент OpenAI-совместимого API: генерация, эмбеддинги, реранкер."""

from __future__ import annotations

import math
import sys
import time

import httpx

from . import config as cfg


class ApiError(RuntimeError):
    pass


def _post(client: httpx.Client, base: str, path: str, key: str, payload: dict, attempts: int = 3) -> dict:
    if not base or not key:
        raise ApiError("Не заданы адрес API или ключ")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.post(base + path, headers={"Authorization": "Bearer " + key}, json=payload)
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                break  # ошибка запроса — повтор не поможет
            time.sleep(1.5 * (attempt + 1))
    raise ApiError(f"{path}: {type(last).__name__}: {last}")


def make_client() -> httpx.Client:
    return httpx.Client(timeout=cfg.HTTP_TIMEOUT)


def chat(client: httpx.Client, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str:
    payload = {
        "model": cfg.LLM_MODEL,
        "messages": messages,
        "temperature": cfg.TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or cfg.MAX_TOKENS,
    }
    for attempt in range(2):
        result = _post(client, cfg.API_BASE, "/chat/completions", cfg.LLM_KEY, payload)
        try:
            content = result["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiError(f"Некорректный ответ модели: {exc}") from exc
        if content.strip():
            return content
        print(f"[llm] пустой ответ, попытка {attempt + 1}", file=sys.stderr)
    raise ApiError("Модель вернула пустой ответ")


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed(client: httpx.Client, texts: list[str], batch: int = 32) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(texts), batch):
        part = texts[start:start + batch]
        result = _post(client, cfg.EMBED_BASE, "/embeddings", cfg.EMBED_KEY,
                       {"model": cfg.EMB_MODEL, "input": part, "encoding_format": "float"})
        vectors: list[list[float] | None] = [None] * len(part)
        for pos, item in enumerate(result["data"]):
            idx = int(item.get("index", pos))
            vectors[idx] = _normalize([float(x) for x in item["embedding"]])
        if any(v is None for v in vectors):
            raise ApiError("Получены не все эмбеддинги")
        out.extend(vectors)  # type: ignore[arg-type]
    return out


def rerank(client: httpx.Client, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    result = _post(client, cfg.RERANK_BASE, "/rerank", cfg.RERANK_KEY,
                   {"model": cfg.RERANKER_MODEL, "query": query, "documents": documents,
                    "top_n": min(top_n, len(documents))}, attempts=2)
    items = result.get("results", result.get("data", []))
    ranked = []
    for item in items:
        idx = int(item["index"])
        score = float(item.get("relevance_score", item.get("score", 0.0)))
        if 0 <= idx < len(documents):
            ranked.append((idx, score))
    return ranked

"""Гибридный поиск: BM25 + плотные эмбеддинги (bge-m3) + RRF + опциональный реранкер.

После ранжирования фрагментов контекст собирается на уровне документов: небольшие
внутренние документы подаются целиком, к ним добавляются связанные документы
(другие редакции того же регламента, дополнительные соглашения к договору).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import config as cfg
from . import llm
from .kb import Chunk, Document, KnowledgeBase

# ---------------------------------------------------------------- токенизация

WORD_RE = re.compile(r"[a-zа-я0-9]+")
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня
еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда
кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда
можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя
впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это нам наш нужно
ли ооо""".split())
ENDINGS = sorted("""иями ями ами иях ого его ому ему ыми ими ией ией ость ости остью ений ения ение ениями ениях ами
ой ей ий ый ая яя ое ее ые ие ую юю ов ев ам ям ом ем ах ях ия ью ья ться тся ать ять ить еть уть ет ит ут ют ал ил ыл
ла ло ли ь а я о е ы и у ю й""".split(), key=len, reverse=True)
SYNONYMS = {
    "аванс": "предоплат", "авансом": "предоплат", "предоплат": "аванс",
    "гаранти": "гарантийн", "штраф": "неустойк", "пен": "неустойк",
    "зарплат": "оплат", "увольнен": "расторжен", "испытательн": "испытан",
    "ндс": "налог добавлен стоимост", "ндфл": "налог доход физическ",
}


def stem(word: str) -> str:
    if len(word) <= 4 or word.isdigit():
        return word
    for end in ENDINGS:
        if word.endswith(end) and len(word) - len(end) >= 4:
            word = word[: -len(end)]
            break
    return word[:8]


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    tokens = []
    for w in WORD_RE.findall(text):
        if w in STOP or (len(w) == 1 and not w.isdigit()):
            continue
        tokens.append(stem(w))
    return tokens


def expand_query(tokens: list[str]) -> list[str]:
    extra = []
    for t in tokens:
        for key, value in SYNONYMS.items():
            if t.startswith(key):
                extra.extend(tokenize(value))
    return tokens + extra


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        q = Counter(query)
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            for t, qn in q.items():
                f = tf.get(t)
                if not f:
                    continue
                s += qn * self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


# ---------------------------------------------------------------- кэш эмбеддингов

def _key(text: str) -> str:
    return hashlib.sha1((cfg.EMB_MODEL + "\n" + text).encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, list[float]] = {}
        if path.is_file():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.data = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data), encoding="utf-8")
        except OSError as exc:
            print(f"[search] не удалось сохранить кэш эмбеддингов: {exc}", file=sys.stderr)


# ---------------------------------------------------------------- поиск

@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Retrieved:
    hits: list[Hit]                      # ранжированные фрагменты
    documents: list[tuple[Document, str]]  # (документ, режим: "full" | "chunks")
    doc_chunks: dict[str, list[Chunk]]   # для режима "chunks" — какие фрагменты взяты
    context: str                         # собранный контекст для модели


class Searcher:
    def __init__(self, kb: KnowledgeBase, client: httpx.Client | None = None) -> None:
        self.kb = kb
        self.chunk_texts = [self._index_text(c) for c in kb.chunks]
        self.bm25 = BM25([tokenize(t) for t in self.chunk_texts])
        self.vectors: list[list[float]] | None = None
        self.reranker_ok = cfg.USE_RERANKER
        self._lock = threading.Lock()
        if cfg.USE_EMBEDDINGS and client is not None:
            self._build_vectors(client)

    def _index_text(self, chunk: Chunk) -> str:
        doc = self.kb.by_id[chunk.doc_id]
        extra = " ".join(doc.numbers)
        return f"{doc.title} {extra}\n{chunk.text}"

    def _build_vectors(self, client: httpx.Client) -> None:
        cache = EmbeddingCache(cfg.CACHE_DIR / "embeddings.json")
        missing = [t for t in self.chunk_texts if _key(t) not in cache.data]
        try:
            if missing:
                print(f"[search] расчёт эмбеддингов: {len(missing)} фрагментов", file=sys.stderr)
                for text, vec in zip(missing, llm.embed(client, missing)):
                    cache.data[_key(text)] = vec
                cache.save()
            self.vectors = [cache.data[_key(t)] for t in self.chunk_texts]
        except (llm.ApiError, KeyError, ValueError, TypeError) as exc:
            print(f"[search] эмбеддинги недоступны, только BM25: {exc}", file=sys.stderr)
            self.vectors = None

    # -------------------------------------------------- ранжирование фрагментов

    def rank(self, client: httpx.Client | None, query: str, focus: str = "") -> list[Hit]:
        """query — полный запрос (с историей), focus — последняя реплика пользователя."""
        n = len(self.kb.chunks)
        q_tokens = expand_query(tokenize(query) + tokenize(focus))
        bm = self.bm25.scores(q_tokens)
        rankings = [sorted(range(n), key=lambda i: bm[i], reverse=True)]

        dense: list[float] | None = None
        if self.vectors is not None and client is not None:
            try:
                qv = llm.embed(client, [focus or query, query] if focus and focus != query else [query])
                dense = [max(sum(a * b for a, b in zip(v, chunk_vec)) for v in qv) for chunk_vec in self.vectors]
                rankings.append(sorted(range(n), key=lambda i: dense[i], reverse=True))
            except llm.ApiError as exc:
                print(f"[search] эмбеддинг запроса не получен: {exc}", file=sys.stderr)

        fused: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for pos, idx in enumerate(ranking[:60]):
                fused[idx] += 1.0 / (60 + pos)
        # бонус за явное упоминание номера договора / контрагента
        q_low = (query + " " + focus).lower().replace("ё", "е")
        for idx, chunk in enumerate(self.kb.chunks):
            doc = self.kb.by_id[chunk.doc_id]
            if any(num.lower() in q_low for num in doc.numbers):
                fused[idx] += 0.01
        order = sorted(fused, key=fused.get, reverse=True)
        hits = [Hit(self.kb.chunks[i], fused[i]) for i in order if bm[i] > 0 or (dense and dense[i] > 0.3)]

        if self.reranker_ok and client is not None and len(hits) > 3:
            pool = hits[:25]
            try:
                ranked = llm.rerank(client, focus or query, [h.chunk.text[:2000] for h in pool], cfg.RERANKER_TOP_N)
                if ranked:
                    top = [Hit(pool[i].chunk, 1.0 + s) for i, s in ranked]
                    seen = {h.chunk.chunk_id for h in top}
                    hits = top + [h for h in hits if h.chunk.chunk_id not in seen]
            except llm.ApiError as exc:
                print(f"[search] реранкер отключён: {exc}", file=sys.stderr)
                with self._lock:
                    self.reranker_ok = False
        return hits

    # -------------------------------------------------- сборка контекста

    def related(self, doc: Document) -> list[Document]:
        out = []
        for other in self.kb.documents:
            if other.doc_id == doc.doc_id or other.category != "company":
                continue
            same_group = doc.category == "company" and other.group == doc.group
            shared_number = any(n in other.text[:600] for n in doc.numbers) or any(n in doc.text[:600] for n in other.numbers)
            if same_group or shared_number:
                out.append(other)
        return out

    def retrieve(self, client: httpx.Client | None, query: str, focus: str = "") -> Retrieved:
        hits = self.rank(client, query, focus)
        top = hits[: cfg.TOP_CHUNKS]

        doc_order: list[str] = []
        doc_chunks: dict[str, list[Chunk]] = defaultdict(list)
        for h in top:
            if h.chunk.doc_id not in doc_order:
                doc_order.append(h.chunk.doc_id)
            doc_chunks[h.chunk.doc_id].append(h.chunk)
        # связанные документы (редакции, доп. соглашения)
        for doc_id in list(doc_order):
            for rel in self.related(self.kb.by_id[doc_id]):
                if rel.doc_id not in doc_order:
                    doc_order.append(rel.doc_id)

        parts: list[str] = []
        selected: list[tuple[Document, str]] = []
        used = 0
        for doc_id in doc_order:
            doc = self.kb.by_id[doc_id]
            status = f"\nСтатус: {doc.status}" if doc.status else ""
            head = f"=== [{doc.doc_id}] {doc.title}{status} ==="
            if len(doc.text) <= cfg.SMALL_DOC_CHARS:
                body, mode = doc.text, "full"
            else:
                chunks = sorted(doc_chunks.get(doc_id, []), key=lambda c: int(c.chunk_id.rsplit("-", 1)[1]))
                if not chunks:
                    continue
                body = "\n…\n".join(c.text.split("\n", 1)[1] if "\n" in c.text else c.text for c in chunks)
                body = f"(приведены только найденные фрагменты статьи)\n{body}"
                mode = "chunks"
            block = f"{head}\n{body}"
            if used + len(block) > cfg.MAX_CONTEXT_CHARS and selected:
                continue
            parts.append(block)
            used += len(block)
            selected.append((doc, mode))
        return Retrieved(hits=top, documents=selected, doc_chunks=dict(doc_chunks), context="\n\n".join(parts))

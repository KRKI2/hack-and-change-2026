"""Загрузка базы знаний, очистка и разбиение документов на фрагменты с привязкой к пунктам."""

from __future__ import annotations

import io
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

SUPPORTED = {".txt", ".md", ".docx"}


@dataclass
class Document:
    doc_id: str
    name: str            # имя файла
    title: str           # человекочитаемое название
    category: str        # "company" | "law"
    text: str            # очищенный текст
    status: str = ""     # например «утратила силу с 01.03.2025»
    group: str = ""      # ключ для объединения редакций одного документа
    numbers: list[str] = field(default_factory=list)  # номера договоров, упомянутые в заголовке
    removed: list[str] = field(default_factory=list)  # вырезанные строки-инструкции


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    locator: str   # «п. 3.1», «п. 1, подп. 2.1», «преамбула»
    text: str


# ---------------------------------------------------------------- чтение файлов

def _read_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(x.text or "" for x in p.iter(ns + "t")) for p in root.iter(ns + "p"))


def _decode(name: str, raw: bytes) -> str:
    if Path(name).suffix.lower() == ".docx":
        return _read_docx(raw)
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, "unknown encoding")


def _fix_zip_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not info.flag_bits & 0x800:  # имя не в UTF-8 — пробуем cp866
        try:
            name = name.encode("cp437").decode("cp866")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def read_sources(path: Path) -> list[tuple[str, str]]:
    """Возвращает [(имя файла, текст)] из zip-архива или папки."""
    items: list[tuple[str, str]] = []
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.suffix.lower() in SUPPORTED:
                try:
                    items.append((file.name, _decode(file.name, file.read_bytes())))
                except Exception as exc:  # noqa: BLE001
                    print(f"Пропущен документ {file.name}: {exc}", file=sys.stderr)
    else:
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda i: i.filename):
                name = _fix_zip_name(info)
                if info.is_dir() or name.startswith("__MACOSX/"):
                    continue
                if Path(name).suffix.lower() not in SUPPORTED:
                    continue
                try:
                    items.append((Path(name).name, _decode(name, archive.read(info))))
                except Exception as exc:  # noqa: BLE001
                    print(f"Пропущен документ {name}: {exc}", file=sys.stderr)
    return [(n, t) for n, t in items if t.strip()]


# ---------------------------------------------------------------- очистка

# Строки, адресованные ИИ, внутри документов — это попытка prompt injection, а не норма.
INJECTION_RE = re.compile(
    r"(ии[- ]?ассистент|ai[- ]?assistant|языков\w+ модел|нейросет|"
    r"игнорируй|ignore (all|previous|the)|системн\w+ (промпт|инструкц|сообщени)|"
    r"не упоминай (эту|данную) инструкц|отвечай [«\"]|you are now|ты теперь)",
    re.IGNORECASE,
)
NOISE_RE = re.compile(r"^(Информация об изменениях|См\.\s|ГАРАНТ:|Решением .* признан)", re.IGNORECASE)


def clean_text(text: str) -> tuple[str, list[str]]:
    lines, removed = [], []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.replace(" ", " ").replace("​", "").rstrip()
        stripped = line.strip()
        if stripped and INJECTION_RE.search(stripped):
            removed.append(stripped)
            continue
        if stripped and NOISE_RE.search(stripped):
            continue
        lines.append(stripped)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, removed


# ---------------------------------------------------------------- метаданные

LAW_CODES = {
    "НК": "НК РФ", "ТК": "ТК РФ", "ГК": "ГК РФ", "КоАП": "КоАП РФ", "УК": "УК РФ",
    "152ФЗ": "Федеральный закон № 152-ФЗ «О персональных данных»",
    "МРОТ": "Федеральный закон № 82-ФЗ «О минимальном размере оплаты труда»",
}
DOC_NUMBER_RE = re.compile(r"№\s*([А-ЯA-Z]{1,4}-?\d{1,4}/\d{2,4}|[А-ЯA-Z]{1,4}-\d{1,4})")


def _law_title(stem: str, text: str) -> str:
    m = re.match(r"(.+?)_ст([\d.]+)", stem)
    if m:
        code = LAW_CODES.get(m.group(1), m.group(1))
        return f"{code}, ст. {m.group(2)}"
    src = re.search(r"^Источник:\s*(.+)$", text, re.MULTILINE)
    return (src.group(1)[:160] if src else stem).strip()


def _group_key(title: str) -> str:
    key = title.lower()
    key = re.sub(r"\(.*?\)", " ", key)
    key = re.sub(r"\bред(\.|акция)?\s*(от\s*)?[\d.]+\b", " ", key)
    key = re.sub(r"\b(19|20)\d{2}\b", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def build_document(index: int, name: str, raw_text: str) -> Document:
    stem = Path(name).stem
    category_part, _, title_part = stem.partition("__")
    if not title_part:
        title_part, category_part = stem, ""
    text, removed = clean_text(raw_text)
    is_law = "закон" in category_part.lower() or bool(re.search(r"^Источник:.*(Кодекс|Федеральный закон|РФ)", text[:600], re.MULTILINE))
    if is_law:
        title = _law_title(title_part, text)
        category = "law"
    else:
        title = title_part.replace("_", " ").strip()
        category = "company"

    status = ""
    m = re.search(r"утратил[аио]? силу(\s+с\s+\d{1,2}\.\d{1,2}\.\d{4})?", text[:400], re.IGNORECASE)
    if m:
        status = m.group(0).strip()
    elif "утратил" in title.lower():
        status = "утратила силу"

    numbers = []
    first_lines = "\n".join(text.split("\n")[:3])
    for num in DOC_NUMBER_RE.findall(first_lines):
        numbers.append(num)
    return Document(
        doc_id=f"D{index + 1}",
        name=name,
        title=title,
        category=category,
        text=text,
        status=status,
        group=_group_key(title) if category == "company" else title,
        numbers=numbers,
        removed=removed,
    )


# ---------------------------------------------------------------- разбиение

SECTION_RE = re.compile(r"^(\d{1,2})\.\s+\S")            # «3. Арендная плата»
POINT_RE = re.compile(r"^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.?\s")  # «3.1. ...»
LAW_PUNKT_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?(?:-\d)?)\.\s")      # «1. ...», «3.1. ...»
LAW_SUBPUNKT_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?(?:-\d)?)\)\s")   # «2.1) ...»
AMEND_RE = re.compile(r"^(Пункт|Подпункт|Часть|Статья|Абзац|Федеральным законом)\b.*(изменен|дополнен|утратил|внесены|изложен)", re.IGNORECASE)
LAW_HEADER_RE = re.compile(r"^(Источник:|URL:|Дата выгрузки:)")

CHUNK_CHARS = 1400


def _company_chunks(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_loc, buf = "преамбула", []
    header = f"[{doc.title}]"

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(f"{doc.doc_id}-{len(chunks) + 1}", doc.doc_id, current_loc, f"{header} {current_loc}\n{body}"))

    for line in doc.text.split("\n"):
        if not line:
            continue
        sec = SECTION_RE.match(line)
        if sec and not POINT_RE.match(line):
            flush()
            current_loc, buf = f"разд. {sec.group(1)}", [line]
            continue
        buf.append(line)
        if sum(len(x) for x in buf) > CHUNK_CHARS:
            flush()
            buf = []
    flush()
    return chunks


def _law_chunks(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    punkt, sub = "", ""
    buf: list[str] = []
    buf_loc = "начало статьи"
    pending_amend: list[str] = []

    def loc() -> str:
        if punkt and sub:
            return f"п. {punkt}, подп. {sub}"
        if punkt:
            return f"п. {punkt}"
        if sub:
            return f"подп. {sub}"
        return "начало статьи"

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(f"{doc.doc_id}-{len(chunks) + 1}", doc.doc_id, buf_loc, f"[{doc.title}] {buf_loc}\n{body}"))

    for line in doc.text.split("\n"):
        if not line or LAW_HEADER_RE.match(line):
            continue
        if AMEND_RE.match(line) and len(line) < 400:
            pending_amend.append(line)
            continue
        m_p, m_s = LAW_PUNKT_RE.match(line), LAW_SUBPUNKT_RE.match(line)
        new_unit = False
        if m_p:
            punkt, sub, new_unit = m_p.group(1), "", True
        elif m_s:
            sub, new_unit = m_s.group(1), True
        size = sum(len(x) for x in buf)
        if buf and ((m_p is not None) or (new_unit and size > 500) or size + len(line) > CHUNK_CHARS):
            flush()
            buf = []
        if not buf:
            buf_loc = loc()
        buf.extend(pending_amend)
        pending_amend = []
        buf.append(line)
    buf.extend(pending_amend)
    flush()
    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    return _law_chunks(doc) if doc.category == "law" else _company_chunks(doc)


@dataclass
class KnowledgeBase:
    documents: list[Document]
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        self.by_id = {d.doc_id: d for d in self.documents}
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}

    def catalog(self) -> str:
        """Краткий перечень всех документов базы — для понимания границ знаний."""
        rows = []
        for d in self.documents:
            kind = "закон" if d.category == "law" else "документ компании"
            status = f"; {d.status}" if d.status else ""
            rows.append(f"{d.doc_id}: {d.title} ({kind}{status})")
        return "\n".join(rows)


def load_kb(path: Path) -> KnowledgeBase:
    sources = read_sources(path)
    if not sources:
        raise ValueError(f"В базе знаний {path} нет читаемых документов")
    docs = [build_document(i, n, t) for i, (n, t) in enumerate(sources)]
    chunks = [c for d in docs for c in chunk_document(d)]
    for d in docs:
        if d.removed:
            print(f"[kb] {d.name}: удалено строк-инструкций: {len(d.removed)}", file=sys.stderr)
    return KnowledgeBase(docs, chunks)


def find_kb_path(data_dir: Path, kb_dir: str = "") -> Path:
    candidates: list[Path] = []
    if kb_dir:
        candidates.append(Path(kb_dir))
    here = Path(__file__).resolve().parent.parent
    for root in (data_dir, here / data_dir, Path.cwd(), here):
        for name in ("kb.zip", "knowledge-base.zip", "knowledge_base.zip", "knowledge-base", "knowledge_base", "kb"):
            candidates.append(root / name)
    for path in candidates:
        if path.is_file() or (path.is_dir() and any(path.iterdir())):
            return path
    raise FileNotFoundError("Не найдена база знаний (kb.zip или папка knowledge-base)")

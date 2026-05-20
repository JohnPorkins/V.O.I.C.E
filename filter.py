"""
Читает JSON из conversation_logs/, выделяет людей, объекты/сущности и факты из текста
диалога, сохраняет сводку в classified_logs/ как «{дата} - classified.json»,
и эмбеддинги в embeddings.db: fact_embeddings и char_embeddings (см. embedding_db.py).

Полная классификация и эмбеддинги: задайте OPENAI_API_KEY (или api_key в .env рядом с проектом).
Без ключа используется грубая эвристика (имена по заглавным буквам и т.п.).
Факты извлекаются только из реплик пользователей, не из ответов ассистента (ai).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import struct
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embedding_db import EMBEDDING_DB_PATH, FACT_TABLE, init_embedding_db

BASE = Path(__file__).resolve().parent
LOGS_DIR = BASE / "conversation_logs"
CLASSIFIED_DIR = BASE / "classified_logs"
CHAT_URL = "https://api.openai.com/v1/chat/completions"
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
MODEL = os.environ.get("OPENAI_CLASSIFY_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _load_env_from_dotenv() -> None:
    dotenv_path = BASE / ".env"
    if not dotenv_path.exists():
        return
    try:
        text = dotenv_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key or not val:
            continue
        if key == "api_key" and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = val
        if key not in os.environ:
            os.environ[key] = val


def _parse_message_line(line: str) -> tuple[str, str]:
    if ":" in line:
        role, text = line.split(":", 1)
        return role.strip(), text.strip()
    return "unknown", line.strip()


def _is_user_role(role: str) -> bool:
    r = role.strip().lower()
    if r in ("ai", "assistant", "system", "bot"):
        return False
    return r == "user" or r.startswith("user_") or r.startswith("user-")


def _heuristic_extract(transcript: str, user_transcript: str) -> dict[str, Any]:
    """Очень грубый офлайн-режим без LLM."""
    people: set[str] = set()
    objects_: set[str] = set()
    facts: list[str] = []

    # Имена вида «Слово Слово» (кириллица/латиница)
    name_like = re.findall(
        r"\b([А-ЯЁІЇЄA-Z][а-яёіїєa-z]+(?:\s+[А-ЯЁІЇЄA-Z][а-яёіїєa-z]+)+)\b",
        transcript,
    )
    for chunk in name_like:
        if len(chunk) < 4:
            continue
        people.add(chunk.strip())

    # Шаблоны «X — Y», «X є Y», «X это Y» — только по тексту пользователей
    for m in re.finditer(
        r"(.{3,80}?)\s*(?:—|–|-|є|это|это|це|це)\s*(.{2,80}?)(?:[.!?]|$)",
        user_transcript,
        re.IGNORECASE,
    ):
        a, b = m.group(1).strip(), m.group(2).strip()
        if len(a) > 2 and len(b) > 2:
            facts.append(f"{a} — {b}".strip())

    return {
        "people": sorted(people),
        "objects": sorted(objects_),
        "facts": facts[:20],
        "mode": "heuristic",
    }


def _classify_openai(transcript: str, user_transcript: str) -> dict[str, Any] | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None

    system = (
        "Ты извлекаешь структурированную информацию из диалога (может быть украинский, русский или смесь). "
        "Верни ТОЛЬКО валидный JSON без markdown:\n"
        '{"people": ["имена и роли людей, о которых говорится в тексте"], '
        '"objects": ["предметы, места, организации, абстрактные сущности — всё, что не человек"], '
        '"facts": ["только из реплик пользователей: короткие утверждения, явно сформулированные или выводимые из них, '
        'например: бананы жёлтые; Максим не любит математику"]}\n'
        "people — только реальные люди/персоны из содержания реплик, не технические префиксы вроде user_011. "
        "objects — не люди. facts — только из блока «Только пользователи» ниже: "
        "короткие факты, явно сформулированные или выводимые только из реплик людей; "
        "никогда не включай в facts содержание реплик ассистента (ai), даже если это звучит как факт. "
        "Если пользователи ничего такого не сказали — facts: []. "
        "people и objects можно брать из полного диалога. "
        "Если категория пуста, используй []."
    )
    user = (
        "Полный диалог (роль: текст):\n\n"
        f"{transcript}\n\n"
        "Только реплики пользователей (не ai), из них извлекай facts:\n\n"
        f"{user_transcript if user_transcript.strip() else '(нет реплик пользователей)'}"
    )

    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return {"error": str(e), "people": [], "objects": [], "facts": [], "mode": "openai_error"}

    try:
        raw = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"error": str(e), "people": [], "objects": [], "facts": [], "mode": "openai_parse_error"}

    out = {
        "people": list(parsed.get("people") or []),
        "objects": list(parsed.get("objects") or []),
        "facts": list(parsed.get("facts") or []),
        "mode": "openai",
    }
    return out


def load_conversation_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def transcript_from_log(doc: dict[str, Any]) -> str:
    messages = doc.get("messages") or []
    lines: list[str] = []
    for line in messages:
        if not isinstance(line, str):
            continue
        role, text = _parse_message_line(line)
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def user_transcript_from_log(doc: dict[str, Any]) -> str:
    messages = doc.get("messages") or []
    lines: list[str] = []
    for line in messages:
        if not isinstance(line, str):
            continue
        role, text = _parse_message_line(line)
        if _is_user_role(role):
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _date_label(path: Path, doc: dict[str, Any]) -> str:
    """Дата для имени файла: YYYY-MM-DD из saved_at или stem исходного лога."""
    saved = doc.get("saved_at")
    if isinstance(saved, str) and saved.strip():
        try:
            dt = datetime.fromisoformat(saved.strip().replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return path.stem


def _serialize_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class FactEmbeddingsDB:
    """Эмбеддинги фактов в embeddings.db → таблица fact_embeddings."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path))
        init_embedding_db(self._conn)

    def close(self) -> None:
        self._conn.close()

    def upsert_facts(
        self,
        facts: list[str],
        vectors: list[list[float]],
        *,
        model: str,
        source_file: str | None,
        classified_file: str | None,
    ) -> int:
        if len(facts) != len(vectors):
            raise ValueError("facts and vectors length mismatch")
        now = datetime.now(timezone.utc).isoformat()
        n = 0
        for text, vec in zip(facts, vectors):
            t = text.strip()
            if not t or not vec:
                continue
            self._conn.execute(
                f"""
                INSERT INTO {FACT_TABLE}
                    (fact_text, embedding, model, source_file, classified_file, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_text, source_file) DO UPDATE SET
                    embedding = excluded.embedding,
                    model = excluded.model,
                    classified_file = excluded.classified_file,
                    created_at = excluded.created_at
                """,
                (
                    t,
                    _serialize_embedding(vec),
                    model,
                    source_file,
                    classified_file,
                    now,
                ),
            )
            n += 1
        self._conn.commit()
        return n


def _embed_texts_openai(texts: list[str]) -> list[list[float]] | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or not texts:
        return None

    body = json.dumps(
        {"model": EMBEDDING_MODEL, "input": texts},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        EMBEDDINGS_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    try:
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]
    except (KeyError, TypeError):
        return None


def store_fact_embeddings(
    facts_db: FactEmbeddingsDB,
    facts: list[str],
    *,
    source_file: str | None,
    classified_file: str | None,
) -> int:
    cleaned = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    if not cleaned:
        return 0

    vectors = _embed_texts_openai(cleaned)
    if vectors is None:
        print("[embeddings] не удалось получить эмбеддинги (нужен OPENAI_API_KEY)")
        return 0

    return facts_db.upsert_facts(
        cleaned,
        vectors,
        model=EMBEDDING_MODEL,
        source_file=source_file,
        classified_file=classified_file,
    )


def _classified_output_path(out_dir: Path, date_label: str, source_stem: str) -> Path:
    """classified_logs/{date} - classified.json; при коллизии — суффикс из stem лога."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{date_label} - classified.json"
    candidate = out_dir / base
    if not candidate.exists():
        return candidate
    return out_dir / f"{date_label} ({source_stem}) - classified.json"


def classify_transcript(transcript: str, user_transcript: str) -> dict[str, Any]:
    if not transcript.strip():
        return {"people": [], "objects": [], "facts": [], "mode": "empty"}
    llm = _classify_openai(transcript, user_transcript)
    if llm is not None and llm.get("mode") == "openai":
        return llm
    if llm is not None and llm.get("error"):
        # при ошибке API всё равно добавим эвристику
        h = _heuristic_extract(transcript, user_transcript)
        h["openai_error"] = llm.get("error")
        h["mode"] = "heuristic_after_error"
        return h
    return _heuristic_extract(transcript, user_transcript)


def run(
    logs_dir: Path,
    out_dir: Path,
    only_name: str | None,
    db_path: Path = EMBEDDING_DB_PATH,
) -> None:
    _load_env_from_dotenv()
    if not logs_dir.is_dir():
        raise SystemExit(f"Папка не найдена: {logs_dir}")

    facts_db = FactEmbeddingsDB(db_path.resolve())
    total_embedded = 0

    json_files = sorted(logs_dir.glob("*.json"))
    if only_name:
        json_files = [p for p in json_files if p.name == only_name]
        if not json_files:
            raise SystemExit(f"Файл {only_name!r} не найден в {logs_dir}")

    written: list[Path] = []
    for path in json_files:
        date_label = path.stem
        try:
            doc = load_conversation_file(path)
        except (json.JSONDecodeError, OSError) as e:
            payload: dict[str, Any] = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": MODEL if os.environ.get("OPENAI_API_KEY") else None,
                "source_file": path.name,
                "error": str(e),
                "people": [],
                "objects": [],
                "facts": [],
            }
        else:
            date_label = _date_label(path, doc)
            transcript = transcript_from_log(doc)
            user_tr = user_transcript_from_log(doc)
            block = classify_transcript(transcript, user_tr)
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": MODEL if os.environ.get("OPENAI_API_KEY") else None,
                "source_file": path.name,
                "saved_at": doc.get("saved_at"),
                "people": block.get("people", []),
                "objects": block.get("objects", []),
                "facts": block.get("facts", []),
                "classification_mode": block.get("mode"),
            }
            if block.get("openai_error"):
                payload["openai_error"] = block["openai_error"]

        out_path = _classified_output_path(out_dir, date_label, path.stem)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written.append(out_path)
        print(f"Записано: {out_path}")

        facts = payload.get("facts") or []
        if facts and not payload.get("error"):
            n = store_fact_embeddings(
                facts_db,
                facts,
                source_file=path.name,
                classified_file=out_path.name,
            )
            if n:
                total_embedded += n
                print(f"  эмбеддинги фактов в БД: {n}")

    facts_db.close()
    print(f"Готово: {len(written)} файл(ов) в {out_dir}")
    if total_embedded:
        print(f"Всего эмбеддингов фактов в {db_path}: {total_embedded}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Классификация conversation_logs → JSON")
    ap.add_argument(
        "--logs",
        type=Path,
        default=LOGS_DIR,
        help=f"Папка с логами (по умолчанию: {LOGS_DIR})",
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=CLASSIFIED_DIR,
        help=f"Папка для «{{дата}} - classified.json» (по умолчанию: {CLASSIFIED_DIR})",
    )
    ap.add_argument("--file", type=str, default=None, help="Обработать только один файл по имени")
    ap.add_argument(
        "--db",
        type=Path,
        default=EMBEDDING_DB_PATH,
        help=f"SQLite БД (fact_embeddings + char_embeddings), по умолчанию: {EMBEDDING_DB_PATH}",
    )
    args = ap.parse_args()
    run(args.logs.resolve(), args.output_dir.resolve(), args.file, args.db.resolve())


if __name__ == "__main__":
    main()

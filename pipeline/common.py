"""
Shared helpers for pipeline stages.

Every stage reads and writes through these functions, so local paths and
gs:// URIs behave the same everywhere:

  read_bytes / write_bytes     raw bytes, local or gs://
  read_text / write_text       UTF-8 text
  read_json / write_json       one JSON object
  read_jsonl / write_jsonl     one JSON object per line
  read_source                  bytes plus a stable source identifier
  parse_json_records / parse_jsonl_records   parse text into records
  json_safe, get_field         make values JSON-safe; read nested fields
  sha256, utc_now, join_path   hashing, timestamps, path joining

Google Cloud Storage is imported lazily, so stages that only use local paths do
not need google-cloud-storage installed.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def join_path(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def is_gcs_uri(value: str) -> bool:
    return value.startswith("gs://")


def split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid GCS URI: {uri!r}. Expected gs://bucket/object")
    return parsed.netloc, parsed.path.lstrip("/")


def get_gcs_client():
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "GCS support requires google-cloud-storage. "
            "Install it with: pip install google-cloud-storage"
        ) from exc
    return storage.Client()


# ---------------------------------------------------------------------
# Bytes and text
# ---------------------------------------------------------------------

def read_bytes(location: str) -> bytes:
    if is_gcs_uri(location):
        bucket_name, blob_name = split_gcs_uri(location)
        blob = get_gcs_client().bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object does not exist: {location}")
        return blob.download_as_bytes()

    path = Path(location).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path.read_bytes()


def write_bytes(location: str, data: bytes, content_type: str | None = None) -> str:
    """Write bytes and return the location written (the resolved path for local files)."""
    if is_gcs_uri(location):
        bucket_name, blob_name = split_gcs_uri(location)
        blob = get_gcs_client().bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        return location

    path = Path(location).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def read_source(source: str) -> tuple[bytes, str]:
    """Return source bytes and a stable identifier (the URI, or the resolved local path)."""
    data = read_bytes(source)
    identifier = source if is_gcs_uri(source) else str(Path(source).expanduser().resolve())
    return data, identifier


def decode_text(raw: bytes, encoding: str) -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not decode input using encoding {encoding!r}. "
            "Set \"encoding\" in the configuration to the correct encoding."
        ) from exc


def read_text(location: str, encoding: str = "utf-8") -> str:
    return read_bytes(location).decode(encoding)


def write_text(location: str, content: str, content_type: str = "text/plain; charset=utf-8") -> str:
    return write_bytes(location, content.encode("utf-8"), content_type)


# ---------------------------------------------------------------------
# JSON and JSONL
# ---------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Convert numpy scalars, arrays, and other values into plain JSON types."""
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def get_field(record: dict, field: str) -> Any:
    """Resolve nested paths such as prediction.label. Raises KeyError when missing."""
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(field)
        value = value[part]
    return value


def parse_json_records(text: str) -> list[Any]:
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # A single JSON object is one record; nested arrays are not guessed at.
        return [payload]
    return [{"value": payload}]


def parse_jsonl_records(text: str) -> list[Any]:
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
    return records


def read_json(location: str) -> dict:
    value = json.loads(read_text(location))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {location}.")
    return value


def read_jsonl(location: str) -> list[dict]:
    records: list[dict] = []
    text = read_text(location, encoding="utf-8-sig")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object on line {line_number}.")
        records.append(record)
    return records


def write_json(location: str, value: Any) -> None:
    content = json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n"
    write_text(location, content, "application/json; charset=utf-8")


def write_jsonl(location: str, records: list[dict]) -> None:
    content = "".join(
        json.dumps(json_safe(record), ensure_ascii=False) + "\n"
        for record in records
    )
    write_text(location, content, "application/x-ndjson; charset=utf-8")

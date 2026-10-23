"""
Stage 1: Data Ingestion
-----------------------
Ingest one source file, preserve the original, and emit a canonical artifact
plus a manifest for downstream pipeline stages.

Supported formats:
  Structured: CSV, JSON, JSONL/NDJSON, Parquet (requires pandas + pyarrow)
  Unstructured: TXT, Markdown, HTML, PDF (PDF requires pypdf)

Sources and destinations can be local paths or gs://bucket/object paths.
For GCS support, install google-cloud-storage and authenticate with ADC.

Example (local source -> local artifacts):
  python stage_1_sample.py --source ./data/customers.csv --output-dir ./runs/run-001

Example (GCS source -> GCS artifacts):
  python stage_1_sample.py \
    --source gs://my-bucket/input/customers.csv \
    --output-prefix gs://my-bucket/runs/run-001/stage_1

Example (local source -> GCS artifacts, preserving original in raw/):
  python stage_1_sample.py \
    --source ./data/reviews.jsonl \
    --output-prefix gs://my-bucket/runs/run-001/stage_1 \
    --raw-prefix gs://my-bucket/raw

Output contract:
  stage_1/
    canonical.jsonl   # one JSON object per record
    manifest.json     # source, interpretation, counts, and artifact locations
    (original file copied to raw storage or retained in the output directory)

This stage intentionally does not clean, impute, deduplicate, validate, or label
the data. Those responsibilities belong to later stages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mimetypes
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


STRUCTURED_EXTENSIONS = {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = STRUCTURED_EXTENSIONS | TEXT_EXTENSIONS | PDF_EXTENSIONS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def read_source(source: str) -> tuple[bytes, str]:
    """Return source bytes and a stable source identifier."""
    if is_gcs_uri(source):
        bucket_name, blob_name = split_gcs_uri(source)
        client = get_gcs_client()
        blob = client.bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object does not exist: {source}")
        return blob.download_as_bytes(), source

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source file does not exist: {path}")
    return path.read_bytes(), str(path)


def upload_bytes(uri: str, data: bytes, content_type: str | None = None) -> None:
    bucket_name, blob_name = split_gcs_uri(uri)
    client = get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)


def save_bytes(destination: str, data: bytes, content_type: str | None = None) -> str:
    if is_gcs_uri(destination):
        upload_bytes(destination, data, content_type)
        return destination

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def decode_text(raw: bytes, encoding: str) -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not decode input using encoding {encoding!r}. "
            "Pass --encoding with the correct encoding."
        ) from exc


def parse_json_records(text: str) -> list[Any]:
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # Preserve a single JSON object as one record; do not guess nested arrays.
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
            raise ValueError(
                f"Invalid JSON on line {line_number}: {exc.msg}"
            ) from exc
    return records
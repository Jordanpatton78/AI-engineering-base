#!/usr/bin/env python3
"""
Stage 1: Data Ingestion
-----------------------
Ingest one source, preserve the original, and emit a canonical artifact plus a
manifest for downstream pipeline stages.

Supported sources:
  Local paths, gs://bucket/object files, and bq://project.dataset.table tables.

Supported formats:
  Structured: CSV, JSON, JSONL/NDJSON, Parquet (requires pandas + pyarrow)
  Unstructured: TXT, Markdown, HTML (requires beautifulsoup4), PDF (requires pypdf)
  BigQuery tables are read as structured rows and stored as a JSONL snapshot.

Optional configuration JSON (validated before any data is read):
{
  "dataset_name": "customers",
  "id_field": "customer_id",
  "columns": ["customer_id", "age", "status"],
  "task": "classification",
  "schema": {"customer_id": "string", "age": "number", "status": "string"},
  "encoding": "utf-8"
}

  dataset_name  label recorded in the manifest (default: source name)
  id_field      field holding the stable record identifier; every record must have it
  columns       keep only these fields in canonical.jsonl (must exist in the source)
  task          intended downstream task, recorded for later stages
  schema        declared types, recorded for later stages; values are not coerced here
  encoding      text encoding for file sources (default utf-8)

Example (local file -> local artifacts):
  python sample.py --source ./data/customers.csv --output-dir ./runs/run-001/stage_1

Example (GCS file -> GCS artifacts, original copied to its own prefix):
  python sample.py \
    --source gs://my-bucket/input/customers.csv \
    --output-prefix gs://my-bucket/runs/run-001/stage_1 \
    --raw-prefix gs://my-bucket/raw \
    --config ./configs/ingest.json

Example (BigQuery table -> local artifacts):
  python sample.py --source bq://my-project.my_dataset.customers --output-dir ./runs/run-001/stage_1

Output contract:
  stage_1/
    canonical.jsonl   # one JSON object per record, with an _ingestion block
    manifest.json     # status, source hash, interpretation, counts, artifact locations
    raw/<name>        # byte-for-byte copy of the source (BigQuery: JSONL snapshot)

Records keep their original fields. Each canonical record adds an _ingestion
block with its 1-based source_record_index, which later stages use for lineage.

This stage intentionally does not clean, impute, deduplicate, validate, or label
the data. Those responsibilities belong to later stages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from common import (
    utc_now,
    sha256,
    join_path,
    write_bytes,
    read_source,
    decode_text,
    parse_json_records,
    parse_jsonl_records,
)


STRUCTURED_EXTENSIONS = {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = STRUCTURED_EXTENSIONS | TEXT_EXTENSIONS | PDF_EXTENSIONS
SUPPORTED_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "object", "array"}
CONFIG_KEYS = {"dataset_name", "id_field", "columns", "task", "schema", "encoding"}
BQ_TABLE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+){2}$")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def validate_config(config: dict) -> dict:
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")

    if "columns" in config:
        columns = config["columns"]
        if not isinstance(columns, list) or not all(isinstance(c, str) and c for c in columns):
            raise ValueError("columns must be a list of non-empty field names.")
    if "id_field" in config and not (isinstance(config["id_field"], str) and config["id_field"]):
        raise ValueError("id_field must be a non-empty field name.")
    for name, kind in config.get("schema", {}).items():
        if kind not in SUPPORTED_SCHEMA_TYPES:
            raise ValueError(
                f"Unsupported schema type {kind!r} for field {name!r}. "
                f"Choose from: {', '.join(sorted(SUPPORTED_SCHEMA_TYPES))}"
            )
    if "schema" in config and not isinstance(config["schema"], dict):
        raise ValueError("schema must be an object mapping field names to types.")
    if "encoding" in config:
        try:
            "".encode(config["encoding"])
        except LookupError as exc:
            raise ValueError(f"Unknown encoding {config['encoding']!r}.") from exc
    return config


def load_config(location: str | None) -> dict:
    if not location:
        return {}
    raw, _ = read_source(location)
    try:
        config = json.loads(decode_text(raw, "utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration is not valid JSON: {exc.msg}") from exc
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    return validate_config(config)


# ---------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------

def read_bigquery(table_id: str) -> tuple[list[dict], bytes]:
    """Read every row of a BigQuery table. Returns (records, JSONL snapshot bytes)."""
    if not BQ_TABLE_PATTERN.match(table_id):
        raise ValueError(f"BigQuery source must be bq://project.dataset.table; got {table_id!r}")
    from google.cloud import bigquery

    rows = bigquery.Client().list_rows(table_id)
    records = [json.loads(json.dumps(dict(row.items()), default=str)) for row in rows]
    snapshot = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    return records, snapshot.encode("utf-8")


def parse_document(raw: bytes, extension: str, encoding: str) -> tuple[list[dict], str]:
    """Parse source bytes into records. Returns (records, source_kind)."""
    if extension == ".csv":
        reader = csv.DictReader(io.StringIO(decode_text(raw, encoding)))
        return [dict(row) for row in reader], "structured"

    if extension == ".json":
        return normalize(parse_json_records(decode_text(raw, encoding))), "structured"

    if extension in (".jsonl", ".ndjson"):
        return normalize(parse_jsonl_records(decode_text(raw, encoding))), "structured"

    if extension == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Parquet sources require pandas and pyarrow.") from exc
        frame = pd.read_parquet(io.BytesIO(raw))
        return normalize(json.loads(frame.to_json(orient="records", date_format="iso"))), "structured"

    if extension in TEXT_EXTENSIONS:
        text = decode_text(raw, encoding)
        if extension in (".html", ".htm"):
            try:
                from bs4 import BeautifulSoup
            except ImportError as exc:
                raise RuntimeError("HTML sources require beautifulsoup4.") from exc
            text = BeautifulSoup(text, "html.parser").get_text("\n")
        return [{"text": text}], "unstructured"

    if extension in PDF_EXTENSIONS:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF sources require pypdf.") from exc
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return [{"text": text}], "unstructured"

    raise ValueError(f"Unsupported extension {extension!r}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


def normalize(items: list[Any]) -> list[dict]:
    return [item if isinstance(item, dict) else {"value": item} for item in items]


def validate_records(records: list[dict], config: dict, kind: str) -> None:
    """Check configured columns and identifiers against the data before anything is written."""
    columns = config.get("columns")
    if columns and kind == "structured":
        present = set().union(*(record.keys() for record in records)) if records else set()
        missing = [column for column in columns if column not in present]
        if missing:
            raise ValueError(f"Configured column(s) not found in source: {', '.join(missing)}")

    id_field = config.get("id_field")
    if id_field:
        for index, record in enumerate(records, start=1):
            if record.get(id_field) in (None, ""):
                raise ValueError(f"Record {index} is missing id_field {id_field!r}.")


def default_dataset_name(source: str) -> str:
    if source.startswith("bq://"):
        return source.split(".")[-1]
    return Path(source).stem or "dataset"


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    started_at = utc_now()
    config = load_config(args.config)
    run_id = args.run_id or str(uuid.uuid4())
    dataset_name = args.dataset_name or config.get("dataset_name") or default_dataset_name(args.source)
    output_root = (args.output_dir or args.output_prefix).rstrip("/")
    id_field = config.get("id_field")
    columns = config.get("columns")

    if args.source.startswith("bq://"):
        table_id = args.source[len("bq://"):]
        records, raw = read_bigquery(table_id)
        source_uri = args.source
        name = table_id.replace(".", "_") + ".jsonl"
        source_format = "bigquery"
        encoding = None
        kind = "structured"
        source_kind_note = "BigQuery rows are read through the query API; the raw copy is a JSONL snapshot."
    else:
        raw, source_uri = read_source(args.source)
        name = Path(source_uri).name
        extension = Path(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension {extension!r}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        source_format = extension.lstrip(".")
        encoding = config.get("encoding", "utf-8")
        records, kind = parse_document(raw, extension, encoding)
        source_kind_note = None

    validate_records(records, config, kind)
    source_sha = sha256(raw)

    canonical_lines = []
    for index, record in enumerate(records, start=1):
        payload = {column: record.get(column) for column in columns} if columns else dict(record)
        payload["_ingestion"] = {
            "source_record_index": index,
            "source_kind": kind,
            "source_name": name,
            "source_sha256": source_sha,
            "dataset_name": dataset_name,
            "run_id": run_id,
            "record_id": str(record[id_field]) if id_field else None,
            "ingested_at": utc_now(),
        }
        canonical_lines.append(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    raw_root = args.raw_prefix.rstrip("/") if args.raw_prefix else join_path(output_root, "raw")
    raw_location = write_bytes(join_path(raw_root, name), raw)

    # Confirm the source was not changed while it was being read and copied.
    if not args.source.startswith("bq://"):
        after_sha = sha256(read_source(args.source)[0])
        if after_sha != source_sha:
            raise RuntimeError("The source changed while it was being ingested; no manifest was written.")

    canonical_location = write_bytes(join_path(output_root, "canonical.jsonl"),
                                    "".join(canonical_lines).encode("utf-8"),
                                    "application/x-ndjson")

    manifest_location = join_path(output_root, "manifest.json")
    manifest = {
        "stage": "stage_1_ingestion",
        "run_id": run_id,
        "dataset_name": dataset_name,
        "status": "succeeded",
        "started_at": started_at,
        "finished_at": utc_now(),
        "source": {
            "uri": source_uri,
            "name": name,
            "format": source_format,
            "sha256": source_sha,
            "bytes": len(raw),
        },
        "interpretation": {
            "source_kind": kind,
            "format": source_format,
            "encoding": encoding,
            "id_field": id_field,
            "columns": columns,
            "task": config.get("task"),
            "schema": config.get("schema"),
            "original_data_modified": False,
            "note": source_kind_note,
        },
        "counts": {
            "records_ingested": len(records),
            "records_written": len(canonical_lines),
        },
        "artifacts": {
            "canonical": canonical_location,
            "original": raw_location,
            "manifest": manifest_location,
        },
        "errors": [],
    }
    write_bytes(manifest_location, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
               "application/json")

    print(json.dumps({
        "status": "succeeded",
        "run_id": run_id,
        "records_ingested": len(records),
        "canonical": canonical_location,
        "manifest": manifest_location,
    }, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest one source into canonical JSONL.")
    parser.add_argument("--source", required=True,
                        help="Local path, gs://bucket/object, or bq://project.dataset.table.")
    parser.add_argument("--output-dir", help="Local output directory.")
    parser.add_argument("--output-prefix", help="Output directory or gs:// prefix.")
    parser.add_argument("--raw-prefix", help="Where to copy the original (defaults to <output>/raw).")
    parser.add_argument("--config", help="Optional configuration JSON (local or gs://).")
    parser.add_argument("--run-id", help="Run identifier (default: a new UUID).")
    parser.add_argument("--dataset-name", help="Dataset label recorded in the manifest.")
    args = parser.parse_args(argv)
    if bool(args.output_dir) == bool(args.output_prefix):
        parser.error("Specify exactly one of --output-dir or --output-prefix.")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

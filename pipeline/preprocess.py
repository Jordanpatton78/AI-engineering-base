#!/usr/bin/env python3
"""
Stage 2: Data Validation and Preparation
---------------------------------------
Consumes Stage 1's canonical.jsonl + manifest.json, profiles the data, reports
quality issues, and optionally applies explicitly configured preparation rules.

By default, this stage is conservative:
  - reports missing values, duplicate records, required-field gaps, type
    irregularities, and unexpected categorical values
  - does NOT silently drop rows, impute values, or coerce data
  - writes a prepared JSONL artifact that preserves every input record
  - writes a quality report and a manifest describing findings and actions

Supported input:
  - Stage 1 canonical JSONL artifact (with its _ingestion block)
  - Local paths or gs://bucket/object URIs
  - The Stage 1 manifest, which must have status "succeeded"

Outputs:
  prepared.jsonl       # records passed to later stages; each keeps _ingestion
                       # and gains _preparation.input_record_index
  quality_report.json  # findings, counts, examples, and potential impact
  manifest.json        # Stage 2 artifact and lineage metadata

Optional configuration JSON:
{
  "required_columns": ["customer_id"],
  "expected_types": {
    "age": "number",
    "is_active": "boolean",
    "email": "string"
  },
  "allowed_values": {
    "status": ["new", "active", "closed"]
  },
  "duplicate_key_columns": ["customer_id"],
  "missing_value_tokens": ["unknown"],
  "actions": {
    "trim_strings": true,
    "drop_exact_duplicates": false,
    "drop_rows_missing_required": false
  }
}

Action details:
  trim_strings                  trims whitespace in string values and turns values
                                that are now missing tokens into null
  drop_rows_missing_required    removes records with a missing required column
  drop_exact_duplicates         removes later copies of a record (ignoring _ingestion)
  coerce_types                  is reserved and rejected until implemented

Missing tokens are compared case-insensitively after trimming. The defaults are
"", "na", "n/a", "null", "none", and "nan"; missing_value_tokens adds to them.

Expected types: number and integer accept numeric strings, because Stage 1 reads
CSV values as text. Values are not converted; coerce_types is not implemented.

Example (local):
  python preprocess.py \
    --input ./runs/run-001/stage_1/canonical.jsonl \
    --input-manifest ./runs/run-001/stage_1/manifest.json \
    --output-dir ./runs/run-001/stage_2 \
    --config ./configs/preprocess.json

Example (GCS, failing the stage when required or type findings exist):
  python preprocess.py \
    --input gs://my-bucket/runs/run-001/stage_1/canonical.jsonl \
    --input-manifest gs://my-bucket/runs/run-001/stage_1/manifest.json \
    --output-prefix gs://my-bucket/runs/run-001/stage_2 \
    --config ./configs/preprocess.json \
    --fail-on-errors

Exit codes: 0 success, 1 failure (nothing from this run is published), 2 quality
errors found with --fail-on-errors (report and manifest are still written).

This stage does not perform model-specific feature engineering. Keep rules
domain-configurable and preserve source_record_index lineage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from common import (
    utc_now,
    join_path,
    write_bytes,
    read_source,
    decode_text,
    parse_jsonl_records,
)


DEFAULT_MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan"}
SUPPORTED_EXPECTED_TYPES = {"string", "number", "integer", "boolean", "object", "array"}
CONFIG_KEYS = {"required_columns", "expected_types", "allowed_values",
               "duplicate_key_columns", "missing_value_tokens", "actions"}
ACTION_KEYS = {"trim_strings", "drop_exact_duplicates", "drop_rows_missing_required", "coerce_types"}
RESERVED_ACTIONS = {"coerce_types"}
ERROR_CATEGORIES = ("required_column_findings", "type_findings", "allowed_value_findings")
IMPACT_TEXT = {
    "required_column_findings": "Records missing required fields can fail joins, lookups, or model inputs.",
    "type_findings": "Wrong types break numeric statistics and model input encoding.",
    "allowed_value_findings": "Unexpected categories can be mis-encoded or silently grouped downstream.",
    "duplicate_key_findings": "Duplicate keys inflate counts and can leak records across train and test splits.",
    "exact_duplicate_findings": "Exact duplicates over-weight records and overstate metric confidence.",
    "missing_value_findings": "Missing values reduce usable rows and can bias results when systematic.",
}


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def validate_config(config: dict) -> dict:
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")

    for key in ("required_columns", "duplicate_key_columns"):
        value = config.get(key, [])
        if not isinstance(value, list) or not all(isinstance(c, str) and c for c in value):
            raise ValueError(f"{key} must be a list of field names.")

    for column, kind in config.get("expected_types", {}).items():
        if kind not in SUPPORTED_EXPECTED_TYPES:
            raise ValueError(
                f"Unsupported expected type {kind!r} for column {column!r}. "
                f"Choose from: {', '.join(sorted(SUPPORTED_EXPECTED_TYPES))}"
            )

    for column, allowed in config.get("allowed_values", {}).items():
        if not isinstance(allowed, list):
            raise ValueError(f"allowed_values for {column!r} must be a list.")

    tokens = config.get("missing_value_tokens", [])
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise ValueError("missing_value_tokens must be a list of strings.")

    actions = config.get("actions", {})
    if not isinstance(actions, dict):
        raise ValueError("actions must be an object.")
    for name, enabled in actions.items():
        if name not in ACTION_KEYS:
            raise ValueError(f"Unknown action {name!r}. Choose from: {', '.join(sorted(ACTION_KEYS))}")
        if name in RESERVED_ACTIONS and enabled:
            raise ValueError(f"Action {name!r} is not implemented in this stage.")
        if not isinstance(enabled, bool):
            raise ValueError(f"Action {name!r} must be true or false.")
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


def check_input_manifest(location: str | None) -> dict | None:
    if not location:
        return None
    raw, _ = read_source(location)
    manifest = json.loads(decode_text(raw, "utf-8"))
    if manifest.get("status") != "succeeded":
        raise ValueError(f"Input manifest status must be 'succeeded'; got {manifest.get('status')!r}")
    return manifest


def read_records(location: str) -> list[dict]:
    raw, _ = read_source(location)
    records = parse_jsonl_records(decode_text(raw, "utf-8"))
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Line {index} must contain a JSON object.")
    return records


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def is_missing(value: Any, tokens: set[str]) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in tokens


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    return isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()) is not None


def matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return is_number(value)
    if expected == "integer":
        return is_integer(value)
    if expected == "boolean":
        return isinstance(value, bool) or (isinstance(value, str) and value.strip().lower() in {"true", "false"})
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False


def content_key(record: dict) -> str:
    """Exact-duplicate identity: the record's fields, ignoring its ingestion lineage."""
    payload = {key: value for key, value in record.items() if key != "_ingestion"}
    return json.dumps(payload, sort_keys=True, default=str)


def profile(records: list[dict], config: dict, tokens: set[str]) -> dict:
    """Run every check against the input records. Indices are 1-based input positions."""
    required = config.get("required_columns", [])
    key_columns = config.get("duplicate_key_columns", [])

    required_findings, required_errors = [], 0
    for column in required:
        indices = [i for i, r in enumerate(records, 1) if is_missing(r.get(column), tokens)]
        if indices:
            required_findings.append({"column": column, "record_indices": indices, "count": len(indices)})
            required_errors += len(indices)

    type_findings = []
    for column, expected in config.get("expected_types", {}).items():
        bad = [(i, r[column]) for i, r in enumerate(records, 1)
               if column in r and not is_missing(r[column], tokens) and not matches_type(r[column], expected)]
        if bad:
            type_findings.append({
                "column": column,
                "expected_type": expected,
                "record_indices": [i for i, _ in bad],
                "examples": [str(value)[:80] for _, value in bad[:3]],
            })

    allowed_findings = []
    for column, allowed in config.get("allowed_values", {}).items():
        bad = [(i, r[column]) for i, r in enumerate(records, 1)
               if column in r and not is_missing(r[column], tokens) and r[column] not in allowed]
        if bad:
            allowed_findings.append({
                "column": column,
                "record_indices": [i for i, _ in bad],
                "unexpected_values": sorted({str(value)[:80] for _, value in bad}),
            })

    groups: dict[str, list[int]] = defaultdict(list)
    key_values: dict[str, dict] = {}
    if key_columns:
        for i, record in enumerate(records, 1):
            if all(not is_missing(record.get(c), tokens) for c in key_columns):
                key = json.dumps({c: record[c] for c in key_columns}, sort_keys=True, default=str)
                groups[key].append(i)
                key_values[key] = {c: record[c] for c in key_columns}
    duplicate_key_findings = [
        {"key": key_values[key], "record_indices": indices}
        for key, indices in groups.items() if len(indices) > 1
    ]

    seen: dict[str, int] = {}
    exact_findings = []
    for i, record in enumerate(records, 1):
        key = content_key(record)
        if key in seen:
            exact_findings.append({"record_index": i, "duplicate_of": seen[key]})
        else:
            seen[key] = i

    columns = []
    for record in records:
        for column in record:
            if column != "_ingestion" and column not in columns:
                columns.append(column)
    missing_findings = []
    for column in columns:
        indices = [i for i, r in enumerate(records, 1) if is_missing(r.get(column), tokens)]
        if indices:
            missing_findings.append({"column": column, "count": len(indices), "record_indices": indices})

    return {
        "required_column_findings": required_findings,
        "required_errors": required_errors,
        "type_findings": type_findings,
        "allowed_value_findings": allowed_findings,
        "duplicate_key_findings": duplicate_key_findings,
        "exact_duplicate_findings": exact_findings,
        "missing_value_findings": missing_findings,
    }


# ---------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------

def trim_value(value: Any, tokens: set[str]) -> Any:
    if isinstance(value, str):
        trimmed = value.strip()
        return None if trimmed.lower() in tokens else trimmed
    return value


def apply_actions(records: list[dict], config: dict, tokens: set[str]) -> tuple[list[dict], list[dict], list[str], dict]:
    """Return (prepared, dropped, actions_applied, dropped_counts). Records carry input_record_index."""
    actions = config.get("actions", {})
    required = config.get("required_columns", [])
    applied: list[str] = []
    dropped: list[dict] = []
    counts = {"explicitly_dropped_missing_required": 0, "explicitly_dropped_exact_duplicates": 0}

    working = [(i, dict(record)) for i, record in enumerate(records, 1)]

    if actions.get("trim_strings"):
        applied.append("trim_strings")
        working = [
            (i, {key: (value if key == "_ingestion" else trim_value(value, tokens))
                 for key, value in record.items()})
            for i, record in working
        ]

    if actions.get("drop_rows_missing_required") and required:
        applied.append("drop_rows_missing_required")
        kept = []
        for i, record in working:
            missing = [c for c in required if is_missing(record.get(c), tokens)]
            if missing:
                dropped.append({"input_record_index": i, "reason": "missing_required", "columns": missing})
                counts["explicitly_dropped_missing_required"] += 1
            else:
                kept.append((i, record))
        working = kept

    if actions.get("drop_exact_duplicates"):
        applied.append("drop_exact_duplicates")
        seen: dict[str, int] = {}
        kept = []
        for i, record in working:
            key = content_key(record)
            if key in seen:
                dropped.append({"input_record_index": i, "reason": "exact_duplicate", "duplicate_of": seen[key]})
                counts["explicitly_dropped_exact_duplicates"] += 1
            else:
                seen[key] = i
                kept.append((i, record))
        working = kept

    prepared = []
    for i, record in working:
        record["_preparation"] = {"input_record_index": i}
        prepared.append(record)
    return prepared, dropped, applied, counts


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def build_report(run_id: str, records: list[dict], prepared: list[dict], found: dict,
                 dropped: list[dict], applied: list[str], counts: dict) -> dict:
    errors = sum(len(found[name]) for name in ERROR_CATEGORIES)
    warnings = len(found["duplicate_key_findings"]) + len(found["exact_duplicate_findings"]) + len(found["missing_value_findings"])
    status = "failed" if errors else ("passed_with_warnings" if warnings else "passed")

    potential_impact = [
        {"category": name, "impact": IMPACT_TEXT[name], "count": len(found[name])}
        for name in IMPACT_TEXT if found.get(name)
    ]

    return {
        "stage": "stage_2_preparation",
        "run_id": run_id,
        "quality_status": status,
        "summary": {
            "records_read": len(records),
            "records_written": len(prepared),
            "required_column_errors": found["required_errors"],
            "type_error_columns": len(found["type_findings"]),
            "allowed_value_error_columns": len(found["allowed_value_findings"]),
            "columns_with_missing_values": len(found["missing_value_findings"]),
            "exact_duplicate_records": len(found["exact_duplicate_findings"]),
            "duplicate_key_groups": len(found["duplicate_key_findings"]),
            **counts,
        },
        "findings": {
            "required_column_findings": found["required_column_findings"],
            "type_findings": found["type_findings"],
            "allowed_value_findings": found["allowed_value_findings"],
            "duplicate_key_findings": found["duplicate_key_findings"],
            "exact_duplicate_findings": found["exact_duplicate_findings"],
            "missing_value_findings": found["missing_value_findings"],
        },
        "potential_impact": potential_impact,
        "actions_applied": applied,
        "dropped_records": dropped,
    }


def run(args: argparse.Namespace) -> int:
    started_at = utc_now()
    run_id = args.run_id or str(uuid.uuid4())
    output_root = (args.output_dir or args.output_prefix).rstrip("/")

    input_manifest = check_input_manifest(args.input_manifest)
    config = load_config(args.config)
    tokens = DEFAULT_MISSING_TOKENS | {t.strip().lower() for t in config.get("missing_value_tokens", [])}

    records = read_records(args.input)
    found = profile(records, config, tokens)
    prepared, dropped, applied, counts = apply_actions(records, config, tokens)
    report = build_report(run_id, records, prepared, found, dropped, applied, counts)

    prepared_location = write_bytes(join_path(output_root, "prepared.jsonl"),
                                   "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in prepared).encode("utf-8"),
                                   "application/x-ndjson")
    report_location = write_bytes(join_path(output_root, "quality_report.json"),
                                 json.dumps(report, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
                                 "application/json")
    manifest_location = join_path(output_root, "manifest.json")
    manifest = {
        "stage": "stage_2_preparation",
        "run_id": run_id,
        "status": "succeeded",
        "quality_status": report["quality_status"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "input": {
            "uri": args.input,
            "stage_1_run_id": input_manifest.get("run_id") if input_manifest else None,
            "stage_1_dataset": input_manifest.get("dataset_name") if input_manifest else None,
            "records": len(records),
        },
        "config": config,
        "actions_applied": applied,
        "counts": {
            "records_read": len(records),
            "records_written": len(prepared),
            "records_dropped": len(dropped),
        },
        "artifacts": {
            "prepared_jsonl": prepared_location,
            "quality_report": report_location,
            "manifest": manifest_location,
        },
        "errors": [],
    }
    write_bytes(manifest_location, json.dumps(manifest, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
               "application/json")

    print(json.dumps({
        "status": "succeeded",
        "quality_status": report["quality_status"],
        "run_id": run_id,
        "records_written": len(prepared),
        "prepared": prepared_location,
        "report": report_location,
        "manifest": manifest_location,
    }, indent=2))

    if args.fail_on_errors and report["quality_status"] == "failed":
        return 2
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile, report, and optionally prepare canonical records.")
    parser.add_argument("--input", required=True, help="Stage 1 canonical JSONL (local or gs://).")
    parser.add_argument("--input-manifest", help="Stage 1 manifest; must have status 'succeeded'.")
    parser.add_argument("--config", help="Optional preparation configuration JSON (local or gs://).")
    parser.add_argument("--output-dir", help="Local output directory.")
    parser.add_argument("--output-prefix", help="Output directory or gs:// prefix.")
    parser.add_argument("--run-id", help="Run identifier (default: a new UUID).")
    parser.add_argument("--fail-on-errors", action="store_true",
                        help="Exit with code 2 when required, type, or allowed-value findings exist.")
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

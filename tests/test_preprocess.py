"""QRD tests for Stage 2: Data Validation and Preparation.

Traceability: QRD 2.1, 2.2, 2.4, 2.5 (implemented actions), and 2.6.
Imputation, fitted feature extraction, leakage checks, and domain-specific plugin
rules are not implemented by the current sample preprocessing stage.
"""
from __future__ import annotations

import json
from pathlib import Path

from conftest import PREPROCESS


def write_jsonl(path: Path, records: list[dict]):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_stage2(run_preprocess, canonical: Path, manifest: Path, out: Path, config=None, *extra):
    return run_preprocess(canonical, manifest, out, config, *extra)[0]


def test_2_1_a_and_2_2_a_quality_rules_detect_missing_duplicates_types_and_categories(
    tmp_path, ingest_csv, run_preprocess
):
    records = [
        {"customer_id": 1, "age": 25, "status": "active"},
        {"customer_id": 1, "age": "old", "status": "unknown"},
        {"customer_id": 3, "age": None, "status": "closed"},
    ]
    canonical = tmp_path / "seeded_issues.jsonl"
    write_jsonl(canonical, records)
    out = tmp_path / "stage_2"
    config = {
        "required_columns": ["customer_id", "age"],
        "expected_types": {"age": "number"},
        "allowed_values": {"status": ["active", "closed"]},
        "duplicate_key_columns": ["customer_id"],
    }

    result, _, _ = run_preprocess(canonical, ingest_csv["manifest_path"], out, config)

    assert result.returncode == 0, result.stderr
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    findings = report["findings"]
    assert report["summary"]["columns_with_missing_values"] >= 1
    assert report["summary"]["exact_duplicate_records"] == 0
    assert findings["duplicate_key_findings"]
    assert findings["type_findings"][0]["column"] == "age"
    assert findings["allowed_value_findings"][0]["column"] == "status"


def test_2_1_b_clean_dataset_has_no_quality_findings(tmp_path, ingest_csv, run_preprocess):
    canonical = tmp_path / "clean.jsonl"
    write_jsonl(canonical, [
        {"id": 1, "age": 25, "status": "active"},
        {"id": 2, "age": 35, "status": "closed"},
    ])
    out = tmp_path / "out"
    config = {
        "required_columns": ["id", "age", "status"],
        "expected_types": {"id": "integer", "age": "number", "status": "string"},
        "allowed_values": {"status": ["active", "closed"]},
    }

    result, _, _ = run_preprocess(canonical, ingest_csv["manifest_path"], out, config)

    assert result.returncode == 0, result.stderr
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["required_column_errors"] == 0
    assert report["summary"]["type_error_columns"] == 0
    assert report["summary"]["allowed_value_error_columns"] == 0
    assert report["summary"]["columns_with_missing_values"] == 0
    assert report["summary"]["exact_duplicate_records"] == 0


def test_2_2_a_exact_duplicates_and_duplicate_keys_are_reported(tmp_path, ingest_csv, run_preprocess):
    canonical = tmp_path / "duplicates.jsonl"
    write_jsonl(canonical, [
        {"id": "x", "value": 1},
        {"id": "x", "value": 1},
        {"id": "x", "value": 2},
    ])
    out = tmp_path / "out"

    result, _, _ = run_preprocess(
        canonical, ingest_csv["manifest_path"], out,
        {"duplicate_key_columns": ["id"]},
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["exact_duplicate_records"] == 1
    assert report["findings"]["duplicate_key_findings"][0]["record_indices"] == [1, 2, 3]


def test_2_4_a_quality_report_includes_counts_impact_and_findings(tmp_path, ingest_csv, run_preprocess):
    canonical = tmp_path / "missing.jsonl"
    write_jsonl(canonical, [{"id": 1, "email": None}, {"id": 2, "email": "a@example.com"}])
    out = tmp_path / "out"

    result, _, _ = run_preprocess(canonical, ingest_csv["manifest_path"], out, {"required_columns": ["email"]})

    assert result.returncode == 0, result.stderr
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["records_read"] == 2
    assert "potential_impact" in report
    assert "required_column_findings" in report["findings"]
    assert report["summary"]["required_column_errors"] == 1


def test_2_5_a_trim_strings_action_is_applied_only_when_configured(tmp_path, ingest_csv, run_preprocess):
    canonical = tmp_path / "strings.jsonl"
    write_jsonl(canonical, [{"id": 1, "name": " Ada ", "status": " NA "}])

    unchanged_out = tmp_path / "unchanged"
    unchanged_result, _, _ = run_preprocess(canonical, ingest_csv["manifest_path"], unchanged_out, {})
    assert unchanged_result.returncode == 0, unchanged_result.stderr
    assert read_jsonl(unchanged_out / "prepared.jsonl")[0]["name"] == " Ada "

    trimmed_out = tmp_path / "trimmed"
    trimmed_result, _, _ = run_preprocess(
        canonical, ingest_csv["manifest_path"], trimmed_out, {"actions": {"trim_strings": True}}
    )
    assert trimmed_result.returncode == 0, trimmed_result.stderr
    cleaned = read_jsonl(trimmed_out / "prepared.jsonl")[0]
    assert cleaned["name"] == "Ada"
    assert cleaned["status"] is None


def test_2_5_a_drop_exact_duplicates_action_is_applied_when_enabled(tmp_path, ingest_csv, run_preprocess):
    canonical = tmp_path / "duplicates.jsonl"
    write_jsonl(canonical, [{"id": 1, "x": "a"}, {"id": 1, "x": "a"}, {"id": 2, "x": "b"}])
    out = tmp_path / "out"

    result, _, _ = run_preprocess(
        canonical, ingest_csv["manifest_path"], out,
        {"actions": {"drop_exact_duplicates": True}},
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["records_read"] == 3
    assert report["summary"]["records_written"] == 2
    assert report["summary"]["explicitly_dropped_exact_duplicates"] == 1
    assert len(read_jsonl(out / "prepared.jsonl")) == 2


def test_2_5_a_drop_rows_missing_required_action_is_applied_when_enabled(tmp_path, ingest_csv, run_preprocess):
    canonical = tmp_path / "missing_required.jsonl"
    write_jsonl(canonical, [{"id": 1}, {"id": 2, "email": "x@example.com"}])
    out = tmp_path / "out"

    result, _, _ = run_preprocess(
        canonical, ingest_csv["manifest_path"], out,
        {"required_columns": ["email"], "actions": {"drop_rows_missing_required": True}},
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["explicitly_dropped_missing_required"] == 1
    assert len(read_jsonl(out / "prepared.jsonl")) == 1


def test_2_5_a_unimplemented_coercion_is_rejected_by_configuration_validation(tmp_path, ingest_csv, run_preprocess):
    canonical = tmp_path / "records.jsonl"
    write_jsonl(canonical, [{"id": "1"}])
    out = tmp_path / "out"

    result, _, _ = run_preprocess(
        canonical, ingest_csv["manifest_path"], out,
        {"actions": {"coerce_types": True}},
    )

    assert result.returncode == 1
    assert "coerce_types' is not implemented" in result.stderr
    assert not (out / "manifest.json").exists()


def test_2_6_a_and_2_6_b_prepared_records_preserve_ingestion_lineage(tmp_path, ingest_csv, run_preprocess):
    out = tmp_path / "out"
    result, _, _ = run_preprocess(ingest_csv["canonical"], ingest_csv["manifest_path"], out, {})

    assert result.returncode == 0, result.stderr
    prepared = read_jsonl(out / "prepared.jsonl")
    assert len(prepared) == 2
    assert [r["_ingestion"]["source_record_index"] for r in prepared] == [1, 2]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input"]["stage_1_run_id"] == "test-run"
    assert manifest["artifacts"]["prepared_jsonl"].endswith("prepared.jsonl")


def test_2_6_b_original_source_is_not_modified_by_preprocessing(ingest_csv, run_preprocess, tmp_path):
    before = ingest_csv["source"].read_bytes()
    out = tmp_path / "out"
    result, _, _ = run_preprocess(ingest_csv["canonical"], ingest_csv["manifest_path"], out, {})
    assert result.returncode == 0, result.stderr
    assert ingest_csv["source"].read_bytes() == before


def test_stage2_rejects_failed_stage1_manifest(tmp_path, ingest_csv, run_preprocess):
    bad_manifest = tmp_path / "failed_manifest.json"
    manifest = dict(ingest_csv["manifest"])
    manifest["status"] = "failed"
    bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    result, _, _ = run_preprocess(
        ingest_csv["canonical"], bad_manifest, tmp_path / "out", {}
    )

    assert result.returncode == 1
    assert "status must be 'succeeded'" in result.stderr


def test_stage2_rejects_invalid_expected_type_config(tmp_path, ingest_csv, run_preprocess):
    result, _, _ = run_preprocess(
        ingest_csv["canonical"], ingest_csv["manifest_path"], tmp_path / "out",
        {"expected_types": {"age": "timestamp-ish"}},
    )
    assert result.returncode == 1
    assert "Unsupported expected type" in result.stderr


def test_fail_on_errors_returns_nonzero_after_writing_report(tmp_path, ingest_csv, run_preprocess):
    canonical = tmp_path / "bad_type.jsonl"
    write_jsonl(canonical, [{"age": "not-a-number"}])
    out = tmp_path / "out"

    result, _, _ = run_preprocess(
        canonical, ingest_csv["manifest_path"], out,
        {"expected_types": {"age": "number"}}, "--fail-on-errors",
    )

    assert result.returncode == 2
    assert (out / "quality_report.json").exists()
    assert (out / "manifest.json").exists()

"""QRD tests for Stage 1: Data Ingestion.

Traceability: QRD 1.1, 1.2, 1.3 (supported fields only), and 1.5.
Cloud-only BigQuery/GCS connector tests are not included because those connectors
are not implemented in the current stage_1_sample.py.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from conftest import ROOT, STAGE_1


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_1_1_a_csv_import_creates_expected_records_and_manifest(tmp_path, run_cli):
    source = tmp_path / "records.csv"
    source.write_text("id,name\n1,Ada\n2,Linus\n", encoding="utf-8")
    output = tmp_path / "out"

    result = run_cli(STAGE_1, "--source", source, "--output-dir", output, "--run-id", "qr-1-1-a")

    assert result.returncode == 0, result.stderr
    records = read_jsonl(output / "canonical.jsonl")
    assert [{"id": r["id"], "name": r["name"]} for r in records] == [
        {"id": "1", "name": "Ada"}, {"id": "2", "name": "Linus"}
    ]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["counts"]["records_ingested"] == 2
    assert manifest["run_id"] == "qr-1-1-a"


def test_1_1_a_jsonl_import_preserves_fields_and_record_order(tmp_path, run_cli):
    source = tmp_path / "records.jsonl"
    source.write_text('{"id":"a","value":3}\n{"id":"b","value":4}\n', encoding="utf-8")
    output = tmp_path / "out"

    result = run_cli(STAGE_1, "--source", source, "--output-dir", output)

    assert result.returncode == 0, result.stderr
    records = read_jsonl(output / "canonical.jsonl")
    assert [(r["id"], r["value"]) for r in records] == [("a", 3), ("b", 4)]
    assert [r["_ingestion"]["source_record_index"] for r in records] == [1, 2]


def test_1_1_a_parquet_import_when_optional_dependencies_are_available(tmp_path, run_cli):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    source = tmp_path / "records.parquet"
    pd.DataFrame([{"id": 1, "name": "Ada"}, {"id": 2, "name": "Linus"}]).to_parquet(source, index=False)
    output = tmp_path / "out"

    result = run_cli(STAGE_1, "--source", source, "--output-dir", output)

    assert result.returncode == 0, result.stderr
    records = read_jsonl(output / "canonical.jsonl")
    assert len(records) == 2
    assert records[0]["name"] == "Ada"


def test_1_1_b_unsupported_extension_fails_without_success_manifest(tmp_path, run_cli):
    source = tmp_path / "records.xlsx"
    source.write_bytes(b"not an excel file")
    output = tmp_path / "out"

    result = run_cli(STAGE_1, "--source", source, "--output-dir", output)

    assert result.returncode != 0
    assert "Unsupported extension" in result.stderr
    assert not (output / "manifest.json").exists()


def test_1_1_b_malformed_jsonl_fails_without_success_manifest(tmp_path, run_cli):
    source = tmp_path / "broken.jsonl"
    source.write_text('{"id":1}\n{broken json}\n', encoding="utf-8")
    output = tmp_path / "out"

    result = run_cli(STAGE_1, "--source", source, "--output-dir", output)

    assert result.returncode == 1
    assert "Invalid JSON on line 2" in result.stderr
    assert not (output / "manifest.json").exists()


def test_1_2_a_text_and_image_reference_records_are_represented(tmp_path, run_cli):
    text_source = tmp_path / "notes.txt"
    text_source.write_text("A short document.", encoding="utf-8")
    text_output = tmp_path / "text_out"
    text_result = run_cli(STAGE_1, "--source", text_source, "--output-dir", text_output)
    assert text_result.returncode == 0, text_result.stderr
    text_record = read_jsonl(text_output / "canonical.jsonl")[0]
    assert text_record["text"] == "A short document."
    assert text_record["_ingestion"]["source_kind"] == "unstructured"

    image_ref_source = tmp_path / "image_refs.jsonl"
    image_ref_source.write_text(
        '{"record_id":"img-1","image_uri":"gs://demo/images/1.jpg","label":"cat"}\n',
        encoding="utf-8",
    )
    image_output = tmp_path / "image_out"
    image_result = run_cli(STAGE_1, "--source", image_ref_source, "--output-dir", image_output)
    assert image_result.returncode == 0, image_result.stderr
    image_record = read_jsonl(image_output / "canonical.jsonl")[0]
    assert image_record["image_uri"] == "gs://demo/images/1.jpg"
    # This stage preserves image references; it does not decode image content.


def test_1_3_a_dataset_name_and_run_id_configuration_are_recorded(tmp_path, run_cli):
    source = tmp_path / "input.csv"
    source.write_text("id\n1\n", encoding="utf-8")
    output = tmp_path / "out"

    result = run_cli(
        STAGE_1, "--source", source, "--output-dir", output,
        "--dataset-name", "configured-dataset", "--run-id", "configured-run",
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "configured-dataset"
    assert manifest["run_id"] == "configured-run"


def test_1_5_a_source_bytes_remain_unchanged(tmp_path, run_cli):
    source = tmp_path / "source.csv"
    original = b"id,name\n1,Ada\n"
    source.write_bytes(original)
    output = tmp_path / "out"

    result = run_cli(STAGE_1, "--source", source, "--output-dir", output)

    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == original
    preserved_copy = output / "raw" / source.name
    assert preserved_copy.read_bytes() == original


def test_1_5_b_manifest_records_original_source_hash_and_reference(ingest_csv):
    import hashlib
    manifest = ingest_csv["manifest"]
    assert manifest["source"]["sha256"] == hashlib.sha256(ingest_csv["source"].read_bytes()).hexdigest()
    assert manifest["artifacts"]["original"]
    assert manifest["interpretation"]["original_data_modified"] is False


def test_invalid_output_argument_combination_is_rejected(tmp_path, run_cli):
    source = tmp_path / "data.csv"
    source.write_text("id\n1\n", encoding="utf-8")
    result = run_cli(STAGE_1, "--source", source)
    assert result.returncode != 0
    assert "Specify exactly one" in result.stderr

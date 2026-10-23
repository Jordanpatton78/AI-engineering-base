"""Shared fixtures for QRD-linked integration tests of available pipeline stages."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE_1 = ROOT / "pipeline/sample.py"
PREPROCESS = ROOT / "pipeline/preprocess.py"
TRAIN = ROOT / "pipeline/train.py"
LABEL = ROOT / "pipeline/label.py"
INFERENCE = ROOT / "pipeline/inference.py"
COMPUTE_METRICS = ROOT / "pipeline/compute_metrics.py"
ANOMALY = ROOT / "pipeline/anomaly_detection.py"
DASHBOARD = ROOT / "pipeline/dashboard.py"
WORKFLOW = ROOT / "pipeline/workflow.yaml"


def pending(reason: str):
    """Mark a QRD test as written but not implemented yet.

    Pending tests raise NotImplementedError; xfail keeps them visible in the
    summary without failing the run. Remove the marker when the behavior exists.
    """
    return pytest.mark.xfail(reason=reason, raises=NotImplementedError, strict=False)


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def labeled_rows(n: int = 40) -> list[dict]:
    """Small binary classification dataset with a learnable signal."""
    return [
        {"id": i, "x": float(i), "color": "red" if i % 2 else "blue", "label": "yes" if i % 2 else "no"}
        for i in range(n)
    ]


def regression_rows(n: int = 40) -> list[dict]:
    """Small regression dataset where y is a linear function of x."""
    return [{"id": i, "x": float(i), "y": 2.0 * i + 1.0} for i in range(n)]


def find_value(node: Any, key: str) -> Any:
    """Return the first value stored under key anywhere in a nested JSON document, or None."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = find_value(child, key)
        if found is not None:
            return found
    return None


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def workflow_jobs() -> list[str]:
    """Cloud Run job names in the order the workflow calls them."""
    return re.findall(r'job_name: "([^"]+)"', workflow_text())


def workflow_condition_for(job: str) -> str | None:
    """Return the run_* flag guarding a job, or None if its step has no condition."""
    text = workflow_text()
    end = text.index(f'job_name: "{job}"')
    start = text.rfind("job_name:", 0, end)
    segment = text[max(start, 0):end]
    conditions = re.findall(r"condition: \$\{(run_\w+)\}", segment)
    return conditions[-1] if conditions else None


TRAIN_BASE_ARGS = ["--target", "label", "--task", "classification", "--random-state", "0"]
EXAMPLE_EXPERT = {
    "expert": {
        "type": "example_rule",
        "field": "signal",
        "positive_value": "hit",
        "positive_label": "yes",
        "negative_label": "no",
    }
}
CLASSIFICATION_METRICS = {"task": "classification", "target_field": "label", "prediction_field": "prediction"}


def _write_config(tmp_path: Path, name: str, config: dict[str, Any]) -> Path:
    path = tmp_path / f"{name}_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def run_train(run_cli, tmp_path: Path, rows: list[dict], name: str = "run", *extra: str):
    """Run Stage 3 training on rows. Later *extra flags override the defaults above."""
    input_path = write_jsonl(tmp_path / f"{name}_input.jsonl", rows)
    out = tmp_path / name
    out.mkdir()
    result = run_cli(
        TRAIN, "--input", input_path, *TRAIN_BASE_ARGS,
        "--model-path", out / "model.joblib",
        "--metrics", out / "metrics.json",
        "--predictions", out / "predictions.jsonl",
        "--manifest", out / "manifest.json",
        *extra,
    )
    return result, out


def run_label(run_cli, tmp_path: Path, rows: list[dict], name: str = "label", config: dict | None = None, *extra: str):
    """Run Stage 4 pseudo-labeling with the example rule expert unless config is given."""
    input_path = write_jsonl(tmp_path / f"{name}_input.jsonl", rows)
    config_path = _write_config(tmp_path, name, config or EXAMPLE_EXPERT)
    out = tmp_path / name
    out.mkdir()
    result = run_cli(
        LABEL, "--input", input_path, "--config", config_path,
        "--output", out / "labeled.jsonl",
        "--report", out / "label_report.json",
        "--manifest", out / "manifest.json",
        "--label-field", "label", "--id-field", "id",
        *extra,
    )
    return result, out


def run_metrics(run_cli, tmp_path: Path, rows: list[dict], name: str = "metrics", config: dict | None = None, *extra: str):
    """Run the metrics stage on labeled rows that contain both label and prediction fields."""
    input_path = write_jsonl(tmp_path / f"{name}_input.jsonl", rows)
    config_path = _write_config(tmp_path, name, config or CLASSIFICATION_METRICS)
    out = tmp_path / name
    out.mkdir()
    result = run_cli(COMPUTE_METRICS, "--input", input_path, "--config", config_path, "--output-prefix", out, *extra)
    return result, out


def run_anomaly(run_cli, tmp_path: Path, rows: list[dict], name: str = "anomaly", features=("x", "y"), *extra: str):
    """Run Stage 3 anomaly detection with LOF on the given numeric feature columns."""
    input_path = write_jsonl(tmp_path / f"{name}_input.jsonl", rows)
    out = tmp_path / name
    out.mkdir()
    result = run_cli(
        ANOMALY, "--input", input_path, "--feature-columns", *features, "--n-neighbors", "5",
        "--output", out / "anomaly_results.jsonl",
        "--report", out / "anomaly_report.json",
        "--manifest", out / "manifest.json",
        *extra,
    )
    return result, out


@pytest.fixture

def run_cli():
    """Run a project Python script as a subprocess and return its result."""
    def _run(script: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *map(str, args)],
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    return _run


@pytest.fixture

def ingest_csv(tmp_path: Path, run_cli):
    """Create a small CSV, run Stage 1, and return paths plus parsed manifest."""
    source = tmp_path / "customers.csv"
    source.write_text("customer_id,age,status\n1,25,active\n2,40,closed\n", encoding="utf-8")
    output = tmp_path / "stage_1"
    result = run_cli(STAGE_1, "--source", source, "--output-dir", output, "--run-id", "test-run")
    assert result.returncode == 0, result.stderr
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "source": source,
        "output": output,
        "canonical": output / "canonical.jsonl",
        "manifest_path": manifest_path,
        "manifest": manifest,
        "result": result,
    }


@pytest.fixture

def run_preprocess(tmp_path: Path, run_cli):
    """Return a helper that runs Stage 2 against a supplied canonical artifact."""
    def _run(input_path: Path, input_manifest: Path, output_dir: Path, config: dict[str, Any] | None = None, *extra: str):
        args: list[str] = [
            "--input", str(input_path),
            "--input-manifest", str(input_manifest),
            "--output-dir", str(output_dir),
        ]
        config_path = None
        if config is not None:
            config_path = tmp_path / f"config-{len(list(tmp_path.glob('config-*.json')))}.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args.extend(["--config", str(config_path)])
        args.extend(extra)
        result = run_cli(PREPROCESS, *args)
        return result, output_dir, config_path
    return _run

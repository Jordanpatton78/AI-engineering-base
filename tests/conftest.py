"""Shared fixtures for QRD-linked integration tests of available pipeline stages."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE_1 = ROOT / "pipeline/sample.py"
PREPROCESS = ROOT / "pipeline/preprocess.py"


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

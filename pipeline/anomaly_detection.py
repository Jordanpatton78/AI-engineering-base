
#!/usr/bin/env python3
"""
Stage 6: Density-based anomaly detection using Local Outlier Factor (LOF).

Input:
    JSONL records, such as Stage 2 prepared.jsonl.
    Optional reference JSONL representing normal behavior.

Output:
    anomaly_results.jsonl
    anomaly_report.json
    manifest.json

Install:
    pip install scikit-learn numpy

Scoring convention:
    Lower density_score = more anomalous.
    Flag a record when density_score < threshold.

Without a reference dataset:
    LOF detects local outliers within the input dataset.

With a reference dataset:
    LOF novelty detection compares new records against the reference
    distribution. The reference should ideally represent normal behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    sha256,
    join_path,
    read_bytes,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


# ---------------------------------------------------------------------
# Storage helpers: local files and Google Cloud Storage
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------

def is_numeric_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False

    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def select_features(
    records: list[dict],
    configured_columns: list[str] | None,
    reference_records: list[dict] | None = None,
) -> list[str]:
    """
    Select numeric features only.

    If columns are configured, use those columns and reject columns that
    contain no usable numeric values. Missing values are handled separately.
    Otherwise, infer numeric columns from the records.
    """

    if configured_columns:
        columns = list(configured_columns)
    else:
        candidates = set()

        for record in records:
            for key, value in record.items():
                if is_numeric_value(value):
                    candidates.add(key)

        columns = sorted(candidates)

        # In reference mode, use only features numeric in the reference too.
        if reference_records is not None:
            columns = [
                column
                for column in columns
                if any(
                    is_numeric_value(record.get(column))
                    for record in reference_records
                )
            ]

    if not columns:
        raise ValueError(
            "No numeric features found. Configure feature_columns in the "
            "JSON config or provide numeric fields in the input."
        )

    for column in columns:
        has_numeric_value = any(
            is_numeric_value(record.get(column))
            for record in records
        )

        if reference_records is not None:
            has_numeric_value = has_numeric_value and any(
                is_numeric_value(record.get(column))
                for record in reference_records
            )

        if not has_numeric_value:
            raise ValueError(
                f"Feature {column!r} has no usable numeric values."
            )

    return columns


def feature_matrix(
    records: list[dict],
    columns: list[str],
) -> np.ndarray:
    """
    Convert records to a numeric matrix.
    Missing and nonnumeric feature values become NaN and are imputed.
    """

    matrix = []

    for index, record in enumerate(records):
        row = []

        for column in columns:
            value = record.get(column)

            if value is None or isinstance(value, bool):
                row.append(np.nan)
                continue

            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                number = np.nan

            if not np.isfinite(number):
                number = np.nan

            row.append(number)

        matrix.append(row)

    result = np.asarray(matrix, dtype=float)

    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError("Feature matrix is empty or invalid.")

    # SimpleImputer may remove all-missing columns depending on settings
    # and library version, so catch the case explicitly before fitting.
    all_missing = np.all(np.isnan(result), axis=0)

    if np.any(all_missing):
        bad_columns = [
            columns[i] for i in np.where(all_missing)[0]
        ]
        raise ValueError(
            f"These selected features are entirely missing: {bad_columns}"
        )

    return result


def make_preprocessor() -> Pipeline:
    """
    Median imputation plus scaling.

    LOF relies on distances. Scaling prevents large-unit features from
    dominating smaller-unit features merely because of their units.
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    input_bytes = read_bytes(args.input)
    records = read_jsonl(args.input)

    if not records:
        raise ValueError("Input contains no records.")

    reference_records = None
    reference_hash = None

    if args.reference:
        reference_bytes = read_bytes(args.reference)
        reference_hash = sha256(reference_bytes)
        reference_records = read_jsonl(args.reference)

        if len(reference_records) < 3:
            raise ValueError(
                "Reference dataset must contain at least 3 records."
            )

    if args.config:
        config = read_json(args.config)
    else:
        config = {}

    feature_columns = (
        args.feature_columns
        or config.get("feature_columns")
    )

    if feature_columns is not None:
        if not isinstance(feature_columns, list) or not all(
            isinstance(column, str) for column in feature_columns
        ):
            raise ValueError(
                "feature_columns must be a list of column names."
            )

    threshold = (
        args.threshold
        if args.threshold is not None
        else config.get("threshold", -1.5)
    )
    threshold = float(threshold)

    n_neighbors = (
        args.n_neighbors
        if args.n_neighbors is not None
        else config.get("n_neighbors", 20)
    )
    n_neighbors = int(n_neighbors)

    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2.")

    run_id = args.run_id or str(uuid.uuid4())
    started = time.time()

    # Select common numeric features and construct feature matrices.
    columns = select_features(
        records,
        feature_columns,
        reference_records,
    )

    X = feature_matrix(records, columns)

    if reference_records is None:
        if len(records) < 3:
            raise ValueError(
                "At least 3 records are required for local outlier detection."
            )

        effective_neighbors = min(n_neighbors, len(records) - 1)

        preprocessor = make_preprocessor()
        X_scaled = preprocessor.fit_transform(X)

        detector = LocalOutlierFactor(
            n_neighbors=effective_neighbors,
            novelty=False,
            contamination="auto",
            n_jobs=-1,
        )

        predictions = detector.fit_predict(X_scaled)

        # negative_outlier_factor_: values nearer -1 are more typical;
        # lower, more negative values indicate stronger outlier behavior.
        density_scores = detector.negative_outlier_factor_

        scoring_mode = "within_dataset_outlier_detection"

    else:
        reference_X = feature_matrix(reference_records, columns)

        effective_neighbors = min(
            n_neighbors, len(reference_records) - 1
        )

        preprocessor = make_preprocessor()
        reference_scaled = preprocessor.fit_transform(reference_X)
        X_scaled = preprocessor.transform(X)

        detector = LocalOutlierFactor(
            n_neighbors=effective_neighbors,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        )

        detector.fit(reference_scaled)

        # In novelty mode, score_samples returns the opposite LOF score:
        # lower values indicate more abnormal observations.
        density_scores = detector.score_samples(X_scaled)
        predictions = np.where(density_scores < -1.5, -1, 1)

        scoring_mode = "reference_based_novelty_detection"

    output_records = []
    anomaly_count = 0

    for index, original in enumerate(records):
        record = dict(original)
        score = float(density_scores[index])

        # Explicit threshold is the authoritative flagging rule.
        is_anomaly = score < threshold

        if is_anomaly:
            anomaly_count += 1

        record["_anomaly"] = {
            "is_anomaly": bool(is_anomaly),
            "density_score": score,
            "threshold": threshold,
            "method": "LocalOutlierFactor",
            "scoring_mode": scoring_mode,
            "features_used": columns,
            "run_id": run_id,
            "explanation": (
                "Local density is substantially lower than the comparison "
                "neighborhood, according to the configured score threshold."
                if is_anomaly
                else
                "Score did not fall below the configured anomaly threshold."
            ),
            "interpretation_limit": (
                "Statistical outlier flag only; not proof of an error, "
                "fraudulent event, or domain-specific anomaly."
            ),
        }

        output_records.append(record)

    output_path = args.output or join_path(
        args.output_prefix, "anomaly_results.jsonl"
    )
    report_path = args.report or join_path(
        args.output_prefix, "anomaly_report.json"
    )
    manifest_path = args.manifest or join_path(
        args.output_prefix, "manifest.json"
    )

    write_jsonl(output_path, output_records)

    finished = time.time()

    report = {
        "stage": "stage_6_anomaly_detection",
        "status": "completed",
        "run_id": run_id,
        "method": "LocalOutlierFactor",
        "scoring_mode": scoring_mode,
        "input": args.input,
        "input_sha256": sha256(input_bytes),
        "reference": args.reference,
        "reference_sha256": reference_hash,
        "record_count": len(records),
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_count / len(records),
        "threshold": threshold,
        "score_rule": "is_anomaly = density_score < threshold",
        "n_neighbors_requested": n_neighbors,
        "n_neighbors_used": effective_neighbors,
        "feature_columns": columns,
        "score_summary": {
            "minimum": float(np.min(density_scores)),
            "maximum": float(np.max(density_scores)),
            "mean": float(np.mean(density_scores)),
            "median": float(np.median(density_scores)),
        },
        "limitations": [
            "LOF scores are relative anomaly scores, not calibrated probabilities.",
            "Thresholds depend on the dataset, selected features, preprocessing, "
            "and whether a reference dataset is used.",
            "Numeric features are used; categorical and unstructured fields "
            "require appropriate feature engineering or embeddings.",
            "Without a reference dataset, LOF compares points within the input "
            "dataset and can be affected if many input records are anomalous.",
            "A statistical anomaly is not necessarily a bad or harmful record.",
        ],
        "duration_seconds": round(finished - started, 3),
    }

    write_json(report_path, report)

    manifest = {
        "stage": "stage_6_anomaly_detection",
        "status": "succeeded",
        "run_id": run_id,
        "input": {
            "path": args.input,
            "sha256": sha256(input_bytes),
            "record_count": len(records),
        },
        "reference": {
            "path": args.reference,
            "sha256": reference_hash,
        } if args.reference else None,
        "outputs": {
            "anomaly_results": output_path,
            "anomaly_report": report_path,
        },
        "configuration": {
            "method": "LocalOutlierFactor",
            "threshold": threshold,
            "n_neighbors": effective_neighbors,
            "feature_columns": columns,
            "scoring_mode": scoring_mode,
        },
        "counts": {
            "input_records": len(records),
            "anomalies": anomaly_count,
            "non_anomalies": len(records) - anomaly_count,
        },
        "next_stage": "stage_7_dashboard_visualization",
        "created_at_unix": finished,
    }

    write_json(manifest_path, manifest)

    print(json.dumps({
        "status": manifest["status"],
        "output": output_path,
        "report": report_path,
        "manifest": manifest_path,
        "records": len(records),
        "anomalies": anomaly_count,
        "threshold": threshold,
        "features": columns,
    }, indent=2))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect low-density anomalies using Local Outlier Factor."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL, such as Stage 2 prepared.jsonl.",
    )
    parser.add_argument(
        "--reference",
        help="Optional JSONL dataset representing normal behavior.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON configuration file.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="Numeric columns to use. If omitted, infer numeric columns.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Flag records whose density_score is below this value. Default: -1.5.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        help="LOF neighborhood size. Default: 20.",
    )
    parser.add_argument(
        "--output-prefix",
        default="./stage_6",
        help="Output directory or GCS prefix.",
    )
    parser.add_argument("--output", help="Explicit output JSONL path.")
    parser.add_argument("--report", help="Explicit report JSON path.")
    parser.add_argument("--manifest", help="Explicit manifest JSON path.")
    parser.add_argument("--run-id", help="Workflow-assigned run ID.")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
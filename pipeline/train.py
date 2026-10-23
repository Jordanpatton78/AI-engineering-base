
#!/usr/bin/env python3
"""
Stage 3: Supervised model training and evaluation.

Supports:
    - Classification
    - Regression
    - Configurable scikit-learn estimators
    - Automatic train / validation / test splitting
    - Numeric and categorical feature preprocessing
    - Model and preprocessing persistence
    - Evaluation metrics, predictions, and run manifest
    - Local files and Google Cloud Storage (gs://)

Install:
    pip install pandas numpy scikit-learn joblib

Important:
    Splitting happens BEFORE fitting preprocessing or models.
    Validation data is used for model selection.
    Test data is reserved for final evaluation.
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

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    sha256,
    join_path,
    read_bytes,
    write_bytes,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


# ---------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------

def make_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Keep scalar, tabular fields.

    Nested dictionaries, lists, and internal pipeline metadata are excluded.
    Users can transform nested or unstructured data into features upstream.
    """

    rows = []

    for record in records:
        row = {}

        for key, value in record.items():
            if key.startswith("_"):
                continue

            if isinstance(value, (dict, list, tuple, set)):
                continue

            if isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def normalize_target(
    dataframe: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in dataframe.columns:
        raise ValueError(
            f"Target column {target!r} not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = dataframe.copy()
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    # Missing targets cannot be used for supervised learning.
    valid_target = dataframe[target].notna()
    dataframe = dataframe.loc[valid_target].copy()

    if dataframe.empty:
        raise ValueError("No records have a nonmissing target.")

    y = dataframe[target]
    X = dataframe.drop(columns=[target])

    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the target.")

    # Drop columns that are completely empty in the usable dataset.
    empty_columns = [
        column for column in X.columns
        if X[column].isna().all()
    ]

    X = X.drop(columns=empty_columns)

    if X.shape[1] == 0:
        raise ValueError("All feature columns are empty.")

    return X, y


def identify_feature_types(
    X: pd.DataFrame,
    configured_numeric: list[str] | None = None,
    configured_categorical: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    configured_numeric = configured_numeric or []
    configured_categorical = configured_categorical or []

    unknown = (
        set(configured_numeric) | set(configured_categorical)
    ) - set(X.columns)

    if unknown:
        raise ValueError(
            f"Configured feature columns do not exist: {sorted(unknown)}"
        )

    overlap = set(configured_numeric) & set(configured_categorical)

    if overlap:
        raise ValueError(
            f"Features cannot be both numeric and categorical: {sorted(overlap)}"
        )

    numeric = []
    categorical = []

    for column in X.columns:
        if column in configured_numeric:
            numeric.append(column)
        elif column in configured_categorical:
            categorical.append(column)
        elif pd.api.types.is_numeric_dtype(X[column]):
            numeric.append(column)
        else:
            categorical.append(column)

    # Categorical columns are normalized to strings for consistent encoding.
    for column in categorical:
        X[column] = X[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    for column in numeric:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    return numeric, categorical


# ---------------------------------------------------------------------
# Split and preprocessing
# ---------------------------------------------------------------------

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple:
    """
    Split into train, validation, and test sets.

    test_size and validation_size are fractions of the full dataset.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    if test_size + validation_size >= 1:
        raise ValueError("Test and validation fractions must sum to less than 1.")

    stratify = y if task == "classification" else None

    # Stratification requires sufficient examples per class. Fall back to
    # random splitting when class counts make stratification impossible.
    if stratify is not None and stratify.value_counts().min() < 2:
        stratify = None

    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    # Convert full-dataset validation fraction into a fraction of the
    # remaining train-plus-validation subset.
    relative_validation_size = validation_size / (1 - test_size)

    train_stratify = (
        y_train_val
        if task == "classification"
        and y_train_val.value_counts().min() >= 2
        else None
    )

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_validation_size,
            random_state=random_state,
            stratify=train_stratify,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_validation_size,
            random_state=random_state,
        )

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "A train, validation, or test split is empty. "
            "Provide more labeled records or adjust split fractions."
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

def build_model(
    task: str,
    model_name: str,
    random_state: int,
    model_params: dict | None = None,
):
    """
    Create a supported scikit-learn model.

    Supported model names:
        random_forest
        extra_trees
        hist_gradient_boosting
    """

    params = dict(model_params or {})
    params.setdefault("random_state", random_state)

    if model_name == "random_forest":
        if task == "classification":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    if model_name == "extra_trees":
        if task == "classification":
            return ExtraTreesClassifier(**params)
        return ExtraTreesRegressor(**params)

    if model_name == "hist_gradient_boosting":
        if task == "classification":
            return HistGradientBoostingClassifier(**params)
        return HistGradientBoostingRegressor(**params)

    raise ValueError(
        f"Unsupported model {model_name!r}. "
        "Choose random_forest, extra_trees, or hist_gradient_boosting."
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
) -> dict:
    predictions = model.predict(X)

    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y, predictions)),
            "precision_weighted": float(
                precision_score(
                    y, predictions, average="weighted", zero_division=0
                )
            ),
            "recall_weighted": float(
                recall_score(
                    y, predictions, average="weighted", zero_division=0
                )
            ),
            "f1_weighted": float(
                f1_score(
                    y, predictions, average="weighted", zero_division=0
                )
            ),
            "f1_macro": float(
                f1_score(
                    y, predictions, average="macro", zero_division=0
                )
            ),
            "classification_report": classification_report(
                y,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y, predictions
            ).tolist(),
            "confusion_matrix_labels": [
                str(value)
                for value in model.named_steps["model"].classes_
            ],
        }

    mse = mean_squared_error(y, predictions)

    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y, predictions)),
    }


def make_predictions(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    split_name: str,
) -> list[dict]:
    predictions = model.predict(X)
    result = []

    probabilities = None

    if task == "classification" and hasattr(
        model.named_steps["model"], "predict_proba"
    ):
        try:
            probabilities = model.predict_proba(X)
        except (AttributeError, ValueError):
            probabilities = None

    classes = (
        model.named_steps["model"].classes_
        if task == "classification"
        and hasattr(model.named_steps["model"], "classes_")
        else None
    )

    for index, (row_index, prediction) in enumerate(
        zip(X.index, predictions)
    ):
        item = {
            "source_index": str(row_index),
            "split": split_name,
            "actual": y.loc[row_index],
            "prediction": prediction,
        }

        if probabilities is not None and classes is not None:
            item["class_probabilities"] = {
                str(label): float(probabilities[index, class_index])
                for class_index, label in enumerate(classes)
            }

        result.append(item)

    return result


# ---------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    started = time.time()

    input_bytes = read_bytes(args.input)
    input_hash = sha256(input_bytes)
    records = read_jsonl(args.input)

    if args.input_manifest:
        input_manifest = read_json(args.input_manifest)

        if input_manifest.get("status") != "succeeded":
            raise ValueError(
                "Input manifest must have status='succeeded'; "
                f"received {input_manifest.get('status')!r}."
            )

    config = read_json(args.config) if args.config else {}

    task = args.task or config.get("task", "classification")
    target = args.target or config.get("target")

    if task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression.")

    if not target:
        raise ValueError(
            "Specify the target with --target or the config's target field."
        )

    model_name = args.model or config.get("model", "random_forest")
    random_state = (
        args.random_state
        if args.random_state is not None
        else config.get("random_state", 42)
    )
    test_size = (
        args.test_size
        if args.test_size is not None
        else config.get("test_size", 0.15)
    )
    validation_size = (
        args.validation_size
        if args.validation_size is not None
        else config.get("validation_size", 0.15)
    )

    dataframe = make_dataframe(records)

    if dataframe.empty:
        raise ValueError("Input contains no usable tabular features.")

    X, y = normalize_target(dataframe, target)

    if task == "classification" and y.nunique() < 2:
        raise ValueError(
            "Classification requires at least two distinct target classes."
        )

    if task == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid_target = y.notna()
        X = X.loc[valid_target]
        y = y.loc[valid_target]

    X = X.copy()

    numeric_columns, categorical_columns = identify_feature_types(
        X,
        configured_numeric=config.get("numeric_features"),
        configured_categorical=config.get("categorical_features"),
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(
        X,
        y,
        task=task,
        test_size=float(test_size),
        validation_size=float(validation_size),
        random_state=int(random_state),
    )

    # Preprocessing is fitted only on training data. Validation and test
    # data are transformed using the training-fitted preprocessing steps.
    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    estimator = build_model(
        task=task,
        model_name=model_name,
        random_state=int(random_state),
        model_params=config.get("model_params", {}),
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    # Train only on the training split.
    model.fit(X_train, y_train)

    # Use validation data to assess the fitted model before final testing.
    validation_metrics = evaluate_model(
        model, X_val, y_val, task
    )

    # Final test metrics are computed once after training.
    test_metrics = evaluate_model(
        model, X_test, y_test, task
    )

    run_id = args.run_id or str(uuid.uuid4())

    output_prefix = args.output_prefix
    model_path = args.model_path or join_path(
        output_prefix, "model.joblib"
    )
    metrics_path = args.metrics or join_path(
        output_prefix, "metrics.json"
    )
    predictions_path = args.predictions or join_path(
        output_prefix, "test_predictions.jsonl"
    )
    manifest_path = args.manifest or join_path(
        output_prefix, "manifest.json"
    )

    # joblib writes a local file, so persist locally then upload if needed.
    if model_path.startswith("gs://"):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".joblib") as temp:
            joblib.dump(model, temp.name)
            temp.seek(0)
            write_bytes(model_path, Path(temp.name).read_bytes())
    else:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    predictions = make_predictions(
        model, X_test, y_test, task, "test"
    )
    write_jsonl(predictions_path, predictions)

    metrics = {
        "stage": "stage_3_model_training",
        "status": "succeeded",
        "run_id": run_id,
        "task": task,
        "target": target,
        "model": model_name,
        "input": args.input,
        "input_sha256": input_hash,
        "dataset_records": len(X),
        "dropped_missing_target_records": len(records) - len(X),
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "split_fractions_requested": {
            "validation": float(validation_size),
            "test": float(test_size),
        },
        "random_state": int(random_state),
        "features": {
            "numeric": numeric_columns,
            "categorical": categorical_columns,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "limitations": [
            "The test split is held out from model fitting and preprocessing.",
            "No hyperparameter search is performed in this baseline implementation.",
            "Feature inference is designed for scalar tabular data.",
            "Random splitting may be inappropriate for time series, grouped data, "
            "or data with temporal leakage; use a task-specific splitter instead.",
            "Pseudo-labels in the target field are not automatically distinguished "
            "from ground-truth labels; filter them out upstream if needed.",
        ],
        "duration_seconds": round(time.time() - started, 3),
    }

    write_json(metrics_path, metrics)

    manifest = {
        "stage": "stage_3_model_training",
        "status": "succeeded",
        "run_id": run_id,
        "input": {
            "path": args.input,
            "sha256": input_hash,
            "record_count": len(records),
        },
        "outputs": {
            "model": model_path,
            "metrics": metrics_path,
            "test_predictions": predictions_path,
        },
        "configuration": {
            "task": task,
            "target": target,
            "model": model_name,
            "random_state": int(random_state),
            "validation_size": float(validation_size),
            "test_size": float(test_size),
        },
        "split_sizes": metrics["split_sizes"],
        "next_stage": "stage_5_model_evaluation",
        "created_at_unix": time.time(),
    }

    write_json(manifest_path, manifest)

    print(json.dumps({
        "status": "succeeded",
        "run_id": run_id,
        "model": model_path,
        "metrics": metrics_path,
        "test_predictions": predictions_path,
        "manifest": manifest_path,
        "split_sizes": metrics["split_sizes"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }, indent=2, default=str))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scikit-learn model with train/validation/test splits."
    )

    parser.add_argument("--input", required=True, help="Input JSONL dataset.")
    parser.add_argument("--input-manifest", help="Optional upstream manifest.")
    parser.add_argument("--config", help="Optional JSON configuration.")
    parser.add_argument("--target", help="Target column.")
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Prediction task.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
        ],
        help="Model family.",
    )
    parser.add_argument("--random-state", type=int, help="Random seed.")
    parser.add_argument(
        "--test-size",
        type=float,
        help="Fraction of usable records held out for testing.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        help="Fraction of usable records used for validation.",
    )
    parser.add_argument(
        "--output-prefix",
        default="./stage_3",
        help="Output directory or GCS prefix.",
    )
    parser.add_argument("--model-path", help="Model output path.")
    parser.add_argument("--metrics", help="Metrics output path.")
    parser.add_argument("--predictions", help="Test predictions output path.")
    parser.add_argument("--manifest", help="Output manifest path.")
    parser.add_argument("--run-id", help="Workflow-assigned run ID.")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
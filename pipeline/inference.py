
#!/usr/bin/env python3
"""Model-agnostic inference stage for a modular ML pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert common model outputs into JSON-serializable values."""
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )
            records.append(item)
    return records


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(
                json_safe(record), ensure_ascii=False
            ) + "\n")


def read_text_uri(uri: str) -> str:
    """Read local files or gs:// URIs."""
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, _, blob_name = uri[5:].partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Invalid GCS URI: {uri}")
        return storage.Client().bucket(
            bucket_name
        ).blob(blob_name).download_as_text(encoding="utf-8")

    return Path(uri).read_text(encoding="utf-8")


def read_jsonl_uri(uri: str) -> list[dict]:
    return [
        json.loads(line)
        for line in read_text_uri(uri).splitlines()
        if line.strip()
    ]


def write_uri(uri: str, content: str) -> None:
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, _, blob_name = uri[5:].partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Invalid GCS URI: {uri}")
        storage.Client().bucket(
            bucket_name
        ).blob(blob_name).upload_from_string(
            content, content_type="text/plain; charset=utf-8"
        )
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def write_json_uri(uri: str, data: dict) -> None:
    write_uri(uri, json.dumps(json_safe(data), indent=2, ensure_ascii=False))


def write_jsonl_uri(uri: str, records: list[dict]) -> None:
    content = "".join(
        json.dumps(json_safe(r), ensure_ascii=False) + "\n"
        for r in records
    )
    write_uri(uri, content)


def load_config(path: str | None) -> dict:
    return read_json(path) if path else {}


def get_field(record: dict, field: str) -> Any:
    """Read nested fields using dot notation, e.g. customer.text."""
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing required field: {field}")
        value = value[part]
    return value


def make_record_id(record: dict, index: int, id_field: str | None) -> str:
    if id_field:
        try:
            return str(get_field(record, id_field))
        except KeyError:
            pass

    # Deterministic fallback for records with no configured ID.
    raw = json.dumps(record, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"record-{index}-{digest}"


def validate_input_manifest(uri: str | None) -> dict | None:
    if not uri:
        return None

    manifest = json.loads(read_text_uri(uri))
    if manifest.get("status") != "succeeded":
        raise ValueError(
            f"Input manifest status must be 'succeeded'; "
            f"got {manifest.get('status')!r}"
        )
    return manifest


# ---------------------------------------------------------------------
# Common adapter interface
# ---------------------------------------------------------------------

class ModelAdapter(ABC):
    """Every model adapter exposes the same predict_batch interface."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def predict_batch(self, records: list[dict]) -> list[Any]:
        """Return one prediction result per input record."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources when an adapter needs cleanup."""


# ---------------------------------------------------------------------
# 1. scikit-learn adapter
# ---------------------------------------------------------------------

class SklearnAdapter(ModelAdapter):
    def __init__(self, config: dict):
        super().__init__(config)

        import joblib

        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("sklearn adapter requires model_path.")

        if model_path.startswith("gs://"):
            from google.cloud import storage

            bucket_name, _, blob_name = model_path[5:].partition("/")
            data = storage.Client().bucket(
                bucket_name
            ).blob(blob_name).download_as_bytes()

            import io
            self.model = joblib.load(io.BytesIO(data))
        else:
            self.model = joblib.load(model_path)

        self.feature_columns = config.get("feature_columns")
        if not self.feature_columns:
            raise ValueError(
                "sklearn adapter requires feature_columns matching "
                "the features used during training."
            )

    def predict_batch(self, records: list[dict]) -> list[Any]:
        import pandas as pd

        rows = []
        for record in records:
            row = {}
            for column in self.feature_columns:
                row[column] = get_field(record, column)
            rows.append(row)

        X = pd.DataFrame(rows, columns=self.feature_columns)
        predictions = self.model.predict(X)

        results = []
        probabilities = None
        classes = None

        if hasattr(self.model, "predict_proba"):
            try:
                probabilities = self.model.predict_proba(X)
                classes = getattr(self.model, "classes_", None)
                if classes is None and hasattr(self.model, "named_steps"):
                    final_estimator = list(
                        self.model.named_steps.values()
                    )[-1]
                    classes = getattr(final_estimator, "classes_", None)
            except (AttributeError, ValueError, NotImplementedError):
                probabilities = None

        for i, prediction in enumerate(predictions):
            result = {"prediction": json_safe(prediction)}

            if probabilities is not None:
                result["probabilities"] = {
                    str(label): float(prob)
                    for label, prob in zip(classes, probabilities[i])
                }

            results.append(result)

        return results


# ---------------------------------------------------------------------
# 2. Hugging Face adapter
# ---------------------------------------------------------------------

class HuggingFaceAdapter(ModelAdapter):
    """
    Supported modes:
      pipeline             Use transformers.pipeline(task=..., model=...)
      embeddings           AutoModel mean-pooled token embeddings
      sequence_classification
                           AutoModelForSequenceClassification
      text_generation      AutoModelForCausalLM
    """

    def __init__(self, config: dict):
        super().__init__(config)

        import torch
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        self.torch = torch
        self.mode = config.get("mode", "pipeline")
        self.text_field = config.get("text_field", "text")
        self.max_length = int(config.get("max_length", 512))
        self.batch_size = int(config.get("batch_size", 8))
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_name = config.get("model_path") or config.get("model_name")

        if not model_name:
            raise ValueError(
                "Hugging Face adapter requires model_path or model_name."
            )

        if self.mode == "pipeline":
            task = config.get("task")
            if not task:
                raise ValueError(
                    "Hugging Face pipeline mode requires task, "
                    "e.g. text-classification or text-generation."
                )
            self.pipe = pipeline(
                task,
                model=model_name,
                tokenizer=config.get("tokenizer_path", model_name),
                device=0 if str(self.device).startswith("cuda") else -1,
            )
            self.tokenizer = None
            self.model = None

        else:
            self.pipe = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.get("tokenizer_path", model_name)
            )

            if self.mode == "embeddings":
                self.model = AutoModel.from_pretrained(model_name)
            elif self.mode == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
            elif self.mode == "text_generation":
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            else:
                raise ValueError(f"Unsupported Hugging Face mode: {self.mode}")

            self.model.to(self.device)
            self.model.eval()

    def predict_batch(self, records: list[dict]) -> list[Any]:
        texts = [str(get_field(r, self.text_field)) for r in records]

        if self.mode == "pipeline":
            kwargs = {}
            if self.config.get("max_new_tokens") is not None:
                kwargs["max_new_tokens"] = int(
                    self.config["max_new_tokens"]
                )
            raw = self.pipe(texts, **kwargs)
            return [{"output": json_safe(item)} for item in raw]

        results = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with self.torch.inference_mode():
                outputs = self.model(**encoded)

                if self.mode == "embeddings":
                    hidden = outputs.last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                        min=1
                    )
                    batch_results = [
                        {"embedding": vector}
                        for vector in pooled.cpu().tolist()
                    ]

                elif self.mode == "sequence_classification":
                    probs = self.torch.softmax(outputs.logits, dim=-1)
                    labels = getattr(self.model.config, "id2label", {})
                    batch_results = []
                    for row_probs, row_logits in zip(
                        probs.cpu().tolist(), outputs.logits.cpu().tolist()
                    ):
                        best_idx = max(
                            range(len(row_probs)),
                            key=lambda i: row_probs[i],
                        )
                        batch_results.append({
                            "label": labels.get(best_idx, str(best_idx)),
                            "score": row_probs[best_idx],
                            "probabilities": {
                                labels.get(i, str(i)): p
                                for i, p in enumerate(row_probs)
                            },
                            "logits": row_logits,
                        })

                elif self.mode == "text_generation":
                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=int(
                            self.config.get("max_new_tokens", 128)
                        ),
                        do_sample=bool(self.config.get("do_sample", False)),
                        **self.config.get("generation_params", {}),
                    )
                    batch_results = []
                    for input_ids, output_ids in zip(
                        encoded["input_ids"], generated
                    ):
                        # Return only newly generated tokens for decoder-only models.
                        new_tokens = output_ids[len(input_ids):]
                        text = self.tokenizer.decode(
                            new_tokens, skip_special_tokens=True
                        )
                        batch_results.append({"text": text})
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")

            results.extend(batch_results)

        return results

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# 3. Ollama local LLM adapter
# ---------------------------------------------------------------------

class OllamaAdapter(ModelAdapter):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get("model_name")
        if not self.model_name:
            raise ValueError("Ollama adapter requires model_name.")
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.text_field = config.get("text_field", "text")
        self.system_prompt = config.get("system_prompt", "")
        self.options = config.get("options", {})

    def predict_batch(self, records: list[dict]) -> list[Any]:
        results = []
        for record in records:
            prompt = str(get_field(record, self.text_field))
            payload = {
                "model": self.model_name,
                "stream": False,
                "messages": [
                    *([{"role": "system", "content": self.system_prompt}]
                      if self.system_prompt else []),
                    {"role": "user", "content": prompt},
                ],
                "options": self.options,
            }
            response = http_json(
                f"{self.base_url.rstrip('/')}/api/chat", payload,
                headers=self.config.get("headers", {}),
                timeout=int(self.config.get("timeout_seconds", 120)),
            )
            message = response.get("message", {})
            results.append({
                "text": message.get("content", ""),
                "model": response.get("model", self.model_name),
                "done": response.get("done"),
            })
        return results


# ---------------------------------------------------------------------
# 4. OpenAI-compatible API adapter
# ---------------------------------------------------------------------

class OpenAICompatibleAdapter(ModelAdapter):
    """
    Calls a Chat Completions-compatible endpoint.
    The API key is read from an environment variable, not config text.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get("model_name")
        self.base_url = config.get(
            "base_url", "https://api.openai.com/v1"
        ).rstrip("/")
        self.text_field = config.get("text_field", "text")
        self.system_prompt = config.get("system_prompt", "")

        if not self.model_name:
            raise ValueError(
                "openai_compatible adapter requires model_name."
            )

        key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(key_env)
        if not self.api_key:
            raise ValueError(
                f"Missing API key environment variable: {key_env}"
            )

    def predict_batch(self, records: list[dict]) -> list[Any]:
        results = []
        for record in records:
            prompt = str(get_field(record, self.text_field))
            messages = []
            if self.system_prompt:
                messages.append({
                    "role": "system", "content": self.system_prompt
                })
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model_name,
                "messages": messages,
                **self.config.get("request_params", {}),
            }
            response = http_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=int(self.config.get("timeout_seconds", 120)),
            )

            choices = response.get("choices", [])
            if not choices:
                raise ValueError("API response contains no choices.")

            results.append({
                "text": choices[0].get("message", {}).get("content", ""),
                "model": response.get("model", self.model_name),
                "finish_reason": choices[0].get("finish_reason"),
                "usage": response.get("usage"),
            })

        return results


def http_json(
    url: str,
    payload: dict,
    headers: dict | None = None,
    timeout: int = 120,
) -> dict:
    request_headers = {
        "Content-Type": "application/json",
        **(headers or {}),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {body[:1000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


# ---------------------------------------------------------------------
# 5. User-defined adapter
# ---------------------------------------------------------------------

class CustomAdapter(ModelAdapter):
    """
    Load an adapter class from a Python module.

    Example:
      module: my_project.adapters
      class: MyAdapter

    The class must inherit ModelAdapter and implement predict_batch().
    """

    def __init__(self, config: dict):
        module_name = config.get("module")
        class_name = config.get("class")
        if not module_name or not class_name:
            raise ValueError(
                "custom adapter requires module and class."
            )
        module = importlib.import_module(module_name)
        adapter_class = getattr(module, class_name)
        self.instance = adapter_class(config)

    def predict_batch(self, records: list[dict]) -> list[Any]:
        return self.instance.predict_batch(records)

    def close(self) -> None:
        if hasattr(self.instance, "close"):
            self.instance.close()


def build_adapter(config: dict) -> ModelAdapter:
    adapter_type = config.get("type")
    adapters = {
        "sklearn": SklearnAdapter,
        "huggingface": HuggingFaceAdapter,
        "ollama": OllamaAdapter,
        "openai_compatible": OpenAICompatibleAdapter,
        "custom": CustomAdapter,
    }
    if adapter_type not in adapters:
        raise ValueError(
            f"Unknown adapter type {adapter_type!r}. "
            f"Choose from: {', '.join(adapters)}"
        )
    return adapters[adapter_type](config)


# ---------------------------------------------------------------------
# Inference stage orchestration
# ---------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> int:
    run_id = args.run_id or str(uuid.uuid4())
    started_at = utc_now()
    started = time.time()

    output_prefix = args.output_prefix.rstrip("/")
    output_records_path = f"{output_prefix}/predictions.jsonl"
    output_report_path = f"{output_prefix}/inference_report.json"
    output_manifest_path = f"{output_prefix}/manifest.json"

    report = {
        "stage": "inference",
        "run_id": run_id,
        "started_at": started_at,
        "status": "running",
        "input_uri": args.input,
        "model_config_uri": args.config,
        "counts": {
            "input_records": 0,
            "succeeded": 0,
            "failed": 0,
        },
        "errors": [],
    }

    adapter = None
    try:
        input_manifest = validate_input_manifest(args.input_manifest)
        records = read_jsonl_uri(args.input)
        report["counts"]["input_records"] = len(records)

        config = load_config(args.config)
        adapter_config = config.get("adapter", config)
        adapter = build_adapter(adapter_config)

        batch_size = int(config.get("batch_size", 16))
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        prediction_field = config.get("prediction_field", "prediction")
        include_input = bool(config.get("include_input_record", True))
        continue_on_error = bool(config.get("continue_on_error", True))
        id_field = config.get("id_field")
        predictions = []

        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            try:
                batch_outputs = adapter.predict_batch(batch)
                if len(batch_outputs) != len(batch):
                    raise ValueError(
                        f"Adapter returned {len(batch_outputs)} predictions "
                        f"for {len(batch)} records."
                    )

                for offset, (record, output) in enumerate(
                    zip(batch, batch_outputs)
                ):
                    index = start + offset
                    result = dict(record) if include_input else {}
                    record_id = make_record_id(record, index, id_field)
                    result[prediction_field] = json_safe(output)
                    result["_inference"] = {
                        "run_id": run_id,
                        "record_id": record_id,
                        "status": "succeeded",
                        "adapter_type": adapter_config.get("type"),
                        "model_name": adapter_config.get("model_name")
                            or adapter_config.get("model_path"),
                        "completed_at": utc_now(),
                    }
                    predictions.append(result)
                    report["counts"]["succeeded"] += 1

            except Exception as exc:
                if not continue_on_error:
                    raise

                # A failed batch is retried record-by-record to isolate failures.
                for offset, record in enumerate(batch):
                    index = start + offset
                    try:
                        one_output = adapter.predict_batch([record])
                        if len(one_output) != 1:
                            raise ValueError(
                                "Adapter did not return exactly one result."
                            )
                        output = one_output[0]
                        result = dict(record) if include_input else {}
                        result[prediction_field] = json_safe(output)
                        result["_inference"] = {
                            "run_id": run_id,
                            "record_id": make_record_id(record, index, id_field),
                            "status": "succeeded",
                            "adapter_type": adapter_config.get("type"),
                            "completed_at": utc_now(),
                        }
                        predictions.append(result)
                        report["counts"]["succeeded"] += 1
                    except Exception as record_exc:
                        result = dict(record) if include_input else {}
                        result["_inference"] = {
                            "run_id": run_id,
                            "record_id": make_record_id(record, index, id_field),
                            "status": "failed",
                            "error": str(record_exc),
                            "completed_at": utc_now(),
                        }
                        predictions.append(result)
                        report["counts"]["failed"] += 1
                        report["errors"].append({
                            "record_id": result["_inference"]["record_id"],
                            "error": str(record_exc),
                        })

        write_jsonl_uri(output_records_path, predictions)

        report["finished_at"] = utc_now()
        report["duration_seconds"] = round(time.time() - started, 3)
        report["status"] = (
            "succeeded"
            if report["counts"]["failed"] == 0
            else "completed_with_errors"
        )
        report["output_uri"] = output_records_path
        report["input_manifest"] = input_manifest
        write_json_uri(output_report_path, report)

        manifest = {
            "stage": "inference",
            "run_id": run_id,
            "status": report["status"],
            "started_at": started_at,
            "finished_at": report["finished_at"],
            "inputs": {
                "data": args.input,
                "input_manifest": args.input_manifest,
                "config": args.config,
            },
            "artifacts": {
                "predictions": output_records_path,
                "report": output_report_path,
            },
            "counts": report["counts"],
            "next_stage": config.get("next_stage"),
        }
        write_json_uri(output_manifest_path, manifest)

        print(json.dumps({
            "status": report["status"],
            "run_id": run_id,
            "predictions": output_records_path,
            "report": output_report_path,
            "manifest": output_manifest_path,
            "counts": report["counts"],
        }, indent=2))

        return 0 if report["counts"]["failed"] == 0 else 2

    except Exception as exc:
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["duration_seconds"] = round(time.time() - started, 3)
        report["errors"].append({
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        try:
            write_json_uri(output_report_path, report)
            write_json_uri(output_manifest_path, {
                "stage": "inference",
                "run_id": run_id,
                "status": "failed",
                "started_at": started_at,
                "finished_at": report["finished_at"],
                "errors": report["errors"],
                "artifacts": {"report": output_report_path},
            })
        except Exception:
            pass

        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using a configurable model adapter."
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL file or gs:// URI.")
    parser.add_argument("--input-manifest",
                        help="Optional successful upstream manifest.")
    parser.add_argument("--config", required=True,
                        help="JSON inference configuration.")
    parser.add_argument("--output-prefix", required=True,
                        help="Output directory or gs:// prefix.")
    parser.add_argument("--run-id",
                        help="Optional externally assigned pipeline run ID.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_inference(parse_args()))
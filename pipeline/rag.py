#!/usr/bin/env python3
"""
Retrieval-augmented generation helpers.

Used in two places:

  1. Stage 2 (optional): build a vector index from BigQuery rows and GCS documents.
     Each chunk is embedded with a BERT-family encoder and written to a BigQuery
     table that serves as the vector database. A JSONL copy of the chunks and an
     index manifest are written next to the stage outputs.

       python rag.py index --config ./configs/rag_index.json \
         --run-id run-001 --output-prefix ./runs/run-001/stage_2

  2. Stage 7 (inference.py and agents.py): retrieve the most relevant chunks for a
     question, keep them within a token budget, and render a prompt that asks the
     model to cite chunk IDs.

Index config (rag_index.json):
{
  "sources": [
    {"type": "bigquery", "table": "my-project.my_dataset.records",
     "id_field": "record_id", "text_fields": ["title", "body"]},
    {"type": "gcs", "prefix": "gs://my-bucket/docs/", "text_fields": ["text"]}
  ],
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "device": "cpu",
  "vector_table": "my-project.my_dataset.chunk_vectors",
  "write_disposition": "WRITE_APPEND",
  "chunk_tokens": 256,
  "chunk_overlap": 32,
  "embed_batch_size": 32
}

Retrieval config (the "retrieval" block in an inference or agent config):
{
  "vector_table": "my-project.my_dataset.chunk_vectors",
  "index_manifest": "./runs/run-001/stage_2/index_manifest.json",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "device": "cpu",
  "query_field": "question",
  "top_k": 5,
  "min_similarity": 0.3,
  "max_context_tokens": 1500,
  "require_context": true,
  "use_brute_force": true
}

Token counts are approximated by whitespace-separated words, so budgets are
conservative for most models and do not need a tokenizer at build time.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


RAG_PROMPT_VERSION = "rag-prompt-v1"
SUPPORTED_GCS_SUFFIXES = (".txt", ".md", ".html", ".htm", ".pdf", ".jsonl")
TABLE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+){2}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

RAG_TEMPLATE = (
    "Answer the question using only the context below. Cite the chunk IDs you "
    "rely on in square brackets, for example [chunk-id]. If the context does not "
    "contain the answer, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
)


class RetrievalError(Exception):
    """Raised when an index or retrieval configuration cannot be used."""


# ---------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    return len(text.split())


def split_gcs_uri(uri: str) -> tuple[str, str]:
    bucket_name, _, blob_name = uri[5:].partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket_name, blob_name


def read_text(uri: str) -> str:
    """Read a local file or gs:// object as UTF-8 text."""
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_name = split_gcs_uri(uri)
        return storage.Client().bucket(bucket_name).blob(blob_name).download_as_text(encoding="utf-8")
    return Path(uri).read_text(encoding="utf-8")


def write_text(uri: str, content: str) -> None:
    """Write a local file or gs:// object as UTF-8 text."""
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_name = split_gcs_uri(uri)
        storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
            content, content_type="text/plain; charset=utf-8"
        )
        return
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def validate_table(table_id: str) -> str:
    if not TABLE_PATTERN.match(table_id):
        raise ValueError(f"Table must be project.dataset.table; got {table_id!r}")
    return table_id


def validate_identifier(name: str) -> str:
    if not IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"Column names must be plain identifiers; got {name!r}")
    return name


# ---------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------

def load_bigquery_documents(source: dict) -> list[dict]:
    """One document per row, built from the configured text fields."""
    from google.cloud import bigquery

    table = validate_table(source["table"])
    id_field = validate_identifier(source.get("id_field", "record_id"))
    text_fields = [validate_identifier(f) for f in source.get("text_fields", ["text"])]
    sql = f"SELECT {', '.join([id_field, *text_fields])} FROM `{table}`"

    documents = []
    for row in bigquery.Client().query(sql).result():
        values = dict(row.items())
        text = "\n\n".join(
            str(values[field]) for field in text_fields
            if values.get(field) not in (None, "")
        )
        if text.strip():
            documents.append({
                "source_uri": f"bq://{table}",
                "record_id": str(values[id_field]),
                "text": text,
            })
    return documents


def parse_gcs_object(uri: str, suffix: str, data: bytes, text_fields: list[str]) -> list[dict]:
    if suffix in (".txt", ".md"):
        return [{"source_uri": uri, "record_id": uri, "text": data.decode("utf-8")}]

    if suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup

        text = BeautifulSoup(data, "html.parser").get_text("\n")
        return [{"source_uri": uri, "record_id": uri, "text": text}]

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return [{"source_uri": uri, "record_id": uri, "text": text}]

    if suffix == ".jsonl":
        documents = []
        for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = "\n\n".join(str(item[f]) for f in text_fields if item.get(f) not in (None, ""))
            if text.strip():
                record_id = str(item.get("id", f"{uri}#{line_number}"))
                documents.append({"source_uri": uri, "record_id": record_id, "text": text})
        return documents

    raise ValueError(f"Unsupported document type: {suffix}")


def load_gcs_documents(source: dict) -> tuple[list[dict], list[dict]]:
    """Return (documents, skipped). Unsupported or unparseable objects are listed as skipped."""
    from google.cloud import storage

    bucket_name, prefix = split_gcs_uri(source["prefix"])
    text_fields = [validate_identifier(f) for f in source.get("text_fields", ["text"])]
    client = storage.Client()

    documents: list[dict] = []
    skipped: list[dict] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        uri = f"gs://{bucket_name}/{blob.name}"
        suffix = Path(blob.name).suffix.lower()
        if suffix not in SUPPORTED_GCS_SUFFIXES:
            skipped.append({"source_uri": uri, "reason": "unsupported file type"})
            continue
        try:
            documents.extend(parse_gcs_object(uri, suffix, blob.download_as_bytes(), text_fields))
        except Exception as exc:
            skipped.append({"source_uri": uri, "reason": str(exc)})
    return documents, skipped


# ---------------------------------------------------------------------
# Chunking and embeddings
# ---------------------------------------------------------------------

def chunk_document(document: dict, chunk_tokens: int, overlap: int) -> list[dict]:
    if chunk_tokens < 1 or overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("chunk_tokens must be >= 1 and overlap must be in [0, chunk_tokens).")

    words = document["text"].split()
    step = chunk_tokens - overlap
    chunks = []
    for position, start in enumerate(range(0, len(words), step)):
        text = " ".join(words[start:start + chunk_tokens])
        chunk_id = sha256_text(
            "|".join([document["source_uri"], document["record_id"], str(position), text])
        )[:32]
        chunks.append({
            "chunk_id": chunk_id,
            "source_uri": document["source_uri"],
            "record_id": document["record_id"],
            "chunk_position": position,
            "chunk_text": text,
        })
        if start + chunk_tokens >= len(words):
            break
    return chunks


class Encoder:
    """BERT-family sentence encoder: mean-pooled, L2-normalized embeddings."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def embed(self, texts: list[str]) -> list[list[float]]:
        import torch.nn.functional as F

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(
                texts[start:start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                normalized = F.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.cpu().tolist())
        return vectors


# ---------------------------------------------------------------------
# BigQuery vector store
# ---------------------------------------------------------------------

VECTOR_SCHEMA_FIELDS = [
    ("chunk_id", "STRING", "NULLABLE"),
    ("run_id", "STRING", "NULLABLE"),
    ("source_uri", "STRING", "NULLABLE"),
    ("record_id", "STRING", "NULLABLE"),
    ("chunk_text", "STRING", "NULLABLE"),
    ("chunk_position", "INT64", "NULLABLE"),
    ("embedding", "FLOAT64", "REPEATED"),
    ("embedding_model", "STRING", "NULLABLE"),
    ("indexed_at", "TIMESTAMP", "NULLABLE"),
]


def vector_schema():
    from google.cloud import bigquery

    return [bigquery.SchemaField(name, kind, mode=mode) for name, kind, mode in VECTOR_SCHEMA_FIELDS]


def write_vector_rows(table_id: str, rows: list[dict], write_disposition: str) -> None:
    from google.cloud import bigquery

    validate_table(table_id)
    client = bigquery.Client()
    schema = vector_schema()
    client.create_table(bigquery.Table(table_id, schema=schema), exists_ok=True)
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=write_disposition,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    client.load_table_from_json(rows, table_id, job_config=job_config).result()


def create_vector_index(table_id: str) -> None:
    """Create a BigQuery vector index on the embedding column. Recommended for large tables."""
    from google.cloud import bigquery

    validate_table(table_id)
    ddl = (
        f"CREATE VECTOR INDEX IF NOT EXISTS chunk_embedding_idx ON `{table_id}`(embedding) "
        "OPTIONS(index_type='IVF', distance_type='COSINE')"
    )
    bigquery.Client().query(ddl).result()


def bigquery_vector_search(table_id: str, vector: list[float], top_k: int, run_id: str,
                           use_brute_force: bool) -> list[dict]:
    from google.cloud import bigquery

    validate_table(table_id)
    options = json.dumps({"use_brute_force": use_brute_force})
    sql = f"""
        SELECT base.chunk_id, base.source_uri, base.record_id, base.chunk_text, distance
        FROM VECTOR_SEARCH(
          TABLE `{table_id}`, 'embedding',
          (SELECT @query_embedding AS embedding),
          top_k => {int(top_k)},
          distance_type => 'COSINE',
          options => '{options}')
        WHERE base.run_id = @run_id
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", vector),
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
    ])
    return [dict(row.items()) for row in bigquery.Client().query(sql, job_config=job_config).result()]


# ---------------------------------------------------------------------
# Index build (Stage 2, optional)
# ---------------------------------------------------------------------

def load_sources(config: dict) -> tuple[list[dict], list[dict]]:
    documents: list[dict] = []
    skipped: list[dict] = []
    for source in config.get("sources", []):
        kind = source.get("type")
        if kind == "bigquery":
            documents.extend(load_bigquery_documents(source))
        elif kind == "gcs":
            found, missing = load_gcs_documents(source)
            documents.extend(found)
            skipped.extend(missing)
        else:
            raise ValueError(f"Unknown source type: {kind!r}; use 'bigquery' or 'gcs'.")
    return documents, skipped


def build_index(config: dict, run_id: str, output_prefix: str, encoder: Any = None) -> dict:
    """Chunk, embed, and store the configured sources. Returns the index manifest."""
    model_name = config["embedding_model"]
    encoder = encoder or Encoder(model_name, config.get("device", "cpu"),
                                 int(config.get("embed_batch_size", 32)))

    documents, skipped = load_sources(config)
    chunks: list[dict] = []
    for document in documents:
        chunks.extend(chunk_document(
            document,
            int(config.get("chunk_tokens", 256)),
            int(config.get("chunk_overlap", 32)),
        ))
    if not chunks:
        raise RetrievalError("No indexable content was found in the configured sources.")

    vectors = encoder.embed([chunk["chunk_text"] for chunk in chunks])
    dimension = len(vectors[0])
    indexed_at = utc_now()
    rows = [
        {**chunk, "run_id": run_id, "embedding": vector,
         "embedding_model": model_name, "indexed_at": indexed_at}
        for chunk, vector in zip(chunks, vectors)
    ]

    output_prefix = output_prefix.rstrip("/")
    if config.get("vector_table"):
        write_vector_rows(config["vector_table"], rows, config.get("write_disposition", "WRITE_APPEND"))
        if config.get("create_vector_index"):
            create_vector_index(config["vector_table"])

    chunk_lines = "".join(
        json.dumps({k: v for k, v in chunk.items()}, ensure_ascii=False) + "\n"
        for chunk in chunks
    )
    write_text(f"{output_prefix}/chunks.jsonl", chunk_lines)

    manifest = {
        "stage": "index",
        "run_id": run_id,
        "status": "succeeded",
        "created_at": indexed_at,
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "vector_table": config.get("vector_table"),
        "chunk_tokens": int(config.get("chunk_tokens", 256)),
        "chunk_overlap": int(config.get("chunk_overlap", 32)),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "skipped_sources": skipped,
        "index_version": sha256_text("\n".join(sorted(c["chunk_id"] for c in chunks))),
        "artifacts": {
            "chunks": f"{output_prefix}/chunks.jsonl",
            "index_manifest": f"{output_prefix}/index_manifest.json",
        },
        "prompt_version": RAG_PROMPT_VERSION,
    }
    write_text(f"{output_prefix}/index_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


# ---------------------------------------------------------------------
# Retrieval and prompt assembly
# ---------------------------------------------------------------------

SearchFn = Callable[[str, list[float], int, str, bool], list[dict]]


def select_within_budget(hits: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    """Keep hits in score order until the token budget is full. Returns (kept, dropped)."""
    kept: list[dict] = []
    used = 0
    for position, hit in enumerate(hits):
        cost = approx_tokens(hit["chunk_text"])
        if used + cost > max_tokens:
            return kept, len(hits) - position
        kept.append(hit)
        used += cost
    return kept, 0


def render_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{hit['chunk_id']}] {hit['chunk_text']}" for hit in hits)
    return RAG_TEMPLATE.format(context=context, question=question)


class Retriever:
    """Retrieves context for questions from a BigQuery vector table built by build_index()."""

    def __init__(self, config: dict, encoder: Any = None, searcher: SearchFn | None = None):
        self.table = validate_table(config["vector_table"])
        self.top_k = int(config.get("top_k", 5))
        self.min_similarity = float(config.get("min_similarity", 0.0))
        self.max_context_tokens = int(config.get("max_context_tokens", 1500))
        self.require_context = bool(config.get("require_context", True))
        self.use_brute_force = bool(config.get("use_brute_force", True))
        self.searcher = searcher or bigquery_vector_search

        manifest = json.loads(read_text(config["index_manifest"]))
        if manifest.get("status") != "succeeded":
            raise RetrievalError(f"Index manifest status must be 'succeeded'; got {manifest.get('status')!r}")
        if config.get("embedding_model") != manifest["embedding_model"]:
            raise RetrievalError(
                f"Embedding model mismatch: index was built with {manifest['embedding_model']!r}, "
                f"retrieval is configured with {config.get('embedding_model')!r}."
            )
        self.manifest = manifest
        self.index_version = manifest["index_version"]
        self.run_id = manifest["run_id"]
        self.encoder = encoder or Encoder(
            config["embedding_model"], config.get("device", "cpu")
        )

    def retrieve(self, question: str) -> list[dict]:
        vector = self.encoder.embed([question])[0]
        if len(vector) != self.manifest["embedding_dimension"]:
            raise RetrievalError(
                f"Query embedding has dimension {len(vector)}; "
                f"the index expects {self.manifest['embedding_dimension']}."
            )
        raw_hits = self.searcher(self.table, vector, self.top_k, self.run_id, self.use_brute_force)
        hits = []
        for raw in raw_hits:
            score = round(1.0 - float(raw["distance"]), 6)
            if score < self.min_similarity:
                continue
            hits.append({
                "chunk_id": raw["chunk_id"],
                "source_uri": raw["source_uri"],
                "record_id": raw["record_id"],
                "chunk_text": raw["chunk_text"],
                "score": score,
            })
        return sorted(hits, key=lambda hit: hit["score"], reverse=True)

    def build_prompt(self, question: str) -> dict:
        """Return {"prompt", "retrieval", "error"} for one question. Never fabricates context."""
        hits = self.retrieve(question)
        kept, dropped = select_within_budget(hits, self.max_context_tokens)
        retrieval = {
            "prompt_version": RAG_PROMPT_VERSION,
            "index_version": self.index_version,
            "query_hash": sha256_text(question),
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "retrieved_count": len(hits),
            "chunk_ids": [hit["chunk_id"] for hit in kept],
            "scores": [hit["score"] for hit in kept],
            "dropped_for_budget": dropped,
        }

        if kept:
            return {"prompt": render_prompt(question, kept), "retrieval": retrieval, "error": None}
        if self.require_context:
            return {
                "prompt": None,
                "retrieval": retrieval,
                "error": "No retrieved context met min_similarity and require_context is true.",
            }
        return {
            "prompt": question,
            "retrieval": {**retrieval, "warning": "No context retrieved; answered without retrieved content."},
            "error": None,
        }


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def run_index(args: argparse.Namespace) -> int:
    config = json.loads(read_text(args.config))
    run_id = args.run_id or str(uuid.uuid4())
    manifest = build_index(config, run_id, args.output_prefix)
    print(json.dumps({key: manifest[key] for key in ("status", "run_id", "chunk_count", "document_count")}, indent=2))
    return 0


def run_query(args: argparse.Namespace) -> int:
    config = json.loads(read_text(args.config))
    retriever = Retriever(config)
    outcome = retriever.build_prompt(args.query)
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
    return 0 if outcome["error"] is None else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the RAG vector index.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    index = subcommands.add_parser("index", help="Chunk, embed, and store sources.")
    index.add_argument("--config", required=True, help="Index configuration JSON (local or gs://).")
    index.add_argument("--run-id", help="Run identifier recorded with every chunk.")
    index.add_argument("--output-prefix", required=True, help="Output directory or gs:// prefix.")
    index.set_defaults(handler=run_index)

    query = subcommands.add_parser("query", help="Retrieve context for a question.")
    query.add_argument("--config", required=True, help="Retrieval configuration JSON (local or gs://).")
    query.add_argument("--query", required=True, help="Question to retrieve context for.")
    query.set_defaults(handler=run_query)

    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    try:
        raise SystemExit(parsed.handler(parsed))
    except Exception as exc:
        print(f"RAG failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

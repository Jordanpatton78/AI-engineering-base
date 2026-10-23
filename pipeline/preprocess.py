1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
#!/usr/bin/env python3
"""
Stage 2: Data Validation and Preparation
---------------------------------------
Consumes Stage 1's canonical.jsonl + manifest.json, profiles the data, reports
quality issues, and optionally applies explicitly configured preparation rules.

By default, this stage is conservative:
  - reports missing values, duplicate records, inconsistent row schemas, and
    type/format irregularities
  - does NOT silently drop rows, impute values, or coerce data
  - writes a prepared JSONL artifact that preserves every input record
  - writes a quality report and a manifest describing findings and actions

Supported input:
  - Stage 1 canonical JSONL artifact
  - Local paths or gs://bucket/object URIs

Outputs:
  prepared.jsonl       # records passed to Stage 3
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
  "missing_value_tokens": ["", "NA", "N/A", "null", "None"],
  "actions": {
    "trim_strings": true,
    "drop_exact_duplicates": false,
    "drop_rows_missing_required": false,
    "coerce_types": false
  }
}

Example:
  python preprocess.py \
    --input ./runs/run-001/stage_1/canonical.jsonl \
    --input-manifest ./runs/run-001/stage_1/manifest.json \
    --output-dir ./runs/run-001/stage_2

GCS example:
  python preprocess.py \
    --input gs://my-bucket/runs/run-001/stage_1/canonical.jsonl \
    --input-manifest gs://my-bucket/runs/run-001/stage_1/manifest.json \
    --output-prefix gs://my-bucket/runs/run-001/stage_2 \
    --config ./configs/preprocess.json

This stage does not perform model-specific feature engineering. Keep rules
domain-configurable and preserve source_record_index lineage where available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan"}
SUPPORTED_EXPECTED_TYPES = {"string", "number", "integer", "boolean", "object", "array"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_gcs_uri(value: str) -> bool:
    return value.startswith("gs://")


def split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid GCS URI: {uri!r}; expected gs://bucket/object")
    return parsed.netloc, parsed.path.lstrip("/")


def get_gcs_client():
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "GCS support requires google-cloud-storage. "
            "Install it with: pip install google-cloud-storage"
        ) from exc
    return storage.Client()


def read_bytes(location: str) -> bytes:
    if is_gcs_uri(location):
        bucket_name, blob_name = split_gcs_uri(location)
        blob = get_gcs_client().bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object does not exist: {location}")
        return blob.download_as_bytes()

    path = Path(location).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path.read_bytes()


def write_bytes(location: str, data: bytes, content_type: str) -> str:
    if is_gcs_uri(location):
        bucket_name, blob_name = split_gcs_uri(location)
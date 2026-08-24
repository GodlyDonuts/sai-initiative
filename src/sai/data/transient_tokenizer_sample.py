"""Select a bounded, broad tokenizer sample from a transient PleIAs stream."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_virtual_transient_stream import (
    ENVELOPE_SCHEMA,
)
from sai.data.pleias_virtual_transient_stream import (
    RECEIPT_SCHEMA as SOURCE_RECEIPT_SCHEMA,
)
from sai.data.pleias_virtual_transient_stream import (
    STATUS as SOURCE_STATUS,
)
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-transient-tokenizer-sample-receipt-v1"
STATUS = "complete_nontraining_tokenizer_sample"
SAMPLE_NAME = "sample.jsonl"


class TransientTokenizerSampleError(RuntimeError):
    """The transient source, bounded selection, or output custody differs."""


def _signed_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TransientTokenizerSampleError("transient envelope differs")
    unsigned = {key: item for key, item in value.items() if key != "envelope_sha256"}
    if (
        value.get("schema") != ENVELOPE_SCHEMA
        or value.get("envelope_sha256") != canonical_sha256(unsigned)
        or value.get("tokenization_ready") is not True
        or value.get("training_ready") is not False
        or value.get("corpus_split") not in {"train", "development"}
        or not isinstance(value.get("semantic_curriculum_phase"), str)
        or not value["semantic_curriculum_phase"]
        or not isinstance(value.get("semantic_domains"), list)
        or not value["semantic_domains"]
        or any(
            not isinstance(domain, str) or not domain
            for domain in value["semantic_domains"]
        )
        or isinstance(value.get("semantic_quality_floor_milli"), bool)
        or not isinstance(value.get("semantic_quality_floor_milli"), int)
        or not 0 <= value["semantic_quality_floor_milli"] <= 10_000
    ):
        raise TransientTokenizerSampleError("transient envelope differs")
    value = dict(value)
    value["document"] = normalize_document(value.get("document"))
    return value


def _stratum(envelope: dict[str, Any]) -> str:
    document = envelope["document"]
    mode = "code" if document["source"]["domain"] == "code" else "prose"
    primary_domain = sorted(set(envelope["semantic_domains"]))[0]
    return "::".join((envelope["semantic_curriculum_phase"], primary_domain, mode))


def _rank(envelope: dict[str, Any]) -> int:
    """Prefer higher verified quality, then deterministic identity diversity."""

    quality_penalty = 10_000 - envelope["semantic_quality_floor_milli"]
    identity = envelope["document"]["identity_sha256"]
    tie_breaker = int(
        hashlib.sha256(
            b"sai-tokenizer-sample-v1:" + bytes.fromhex(identity)
        ).hexdigest(),
        16,
    )
    return (quality_penalty << 256) | tie_breaker


def build_sample(
    source: TextIO,
    source_receipt_path: Path,
    output_root: Path,
    *,
    maximum_utf8_bytes: int,
) -> dict[str, Any]:
    """Keep a dynamically rebalanced breadth sample under an exact byte cap."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or isinstance(maximum_utf8_bytes, bool)
        or not isinstance(maximum_utf8_bytes, int)
        or maximum_utf8_bytes < 1_000_000
    ):
        raise TransientTokenizerSampleError("tokenizer sample arguments differ")
    heaps: dict[str, list[tuple[int, str, int, bytes]]] = {}
    heap_bytes: Counter[str] = Counter()
    input_digest = hashlib.sha256()
    input_counts: Counter[str] = Counter()

    def rebalance() -> None:
        cap = maximum_utf8_bytes // len(heaps)
        for key, heap in heaps.items():
            while heap and heap_bytes[key] > cap:
                _negative_rank, _identity, size, _encoded = heapq.heappop(heap)
                heap_bytes[key] -= size

    for line in source:
        encoded_input = line.encode()
        input_digest.update(encoded_input)
        if not line.strip():
            raise TransientTokenizerSampleError("transient stream contains blank rows")
        try:
            envelope = _signed_envelope(json.loads(line))
        except json.JSONDecodeError as error:
            raise TransientTokenizerSampleError(
                "transient stream contains malformed JSON"
            ) from error
        input_counts["documents"] += 1
        input_counts["text_utf8_bytes"] += len(envelope["document"]["text"].encode())
        if envelope["corpus_split"] == "development":
            input_counts["development_documents_excluded"] += 1
            continue
        key = _stratum(envelope)
        if key not in heaps:
            heaps[key] = []
            rebalance()
        document_line = (
            json.dumps(
                envelope["document"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        size = len(document_line)
        cap = maximum_utf8_bytes // len(heaps)
        if size > cap:
            input_counts["oversize_train_documents_excluded"] += 1
            continue
        rank = _rank(envelope)
        identity = envelope["document"]["identity_sha256"]
        heapq.heappush(heaps[key], (-rank, identity, size, document_line))
        heap_bytes[key] += size
        rebalance()

    source_receipt = _load_signed(source_receipt_path, SOURCE_RECEIPT_SCHEMA)
    if (
        source_receipt.get("status") != SOURCE_STATUS
        or source_receipt.get("source_text_persisted_by_compiler") is not False
        or source_receipt.get("ordered_jsonl_sha256") != input_digest.hexdigest()
        or source_receipt.get("counts", {}).get("documents")
        != input_counts["documents"]
        or not input_counts["documents"]
        or not heaps
    ):
        raise TransientTokenizerSampleError("transient source receipt differs")
    selected = []
    for key, heap in heaps.items():
        for negative_rank, identity, size, encoded in heap:
            selected.append((key, -negative_rank, identity, size, encoded))
    selected.sort(key=lambda row: (row[0], row[1], row[2]))
    selected_bytes = sum(row[3] for row in selected)
    if not selected or selected_bytes > maximum_utf8_bytes:
        raise TransientTokenizerSampleError("bounded tokenizer sample differs")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    try:
        sample = stage / SAMPLE_NAME
        output_digest = hashlib.sha256()
        ordered_identities = hashlib.sha256()
        selected_counts: Counter[str] = Counter()
        with sample.open("wb") as handle:
            for key, _rank_value, identity, size, encoded in selected:
                handle.write(encoded)
                output_digest.update(encoded)
                ordered_identities.update(bytes.fromhex(identity))
                selected_counts["documents"] += 1
                selected_counts["jsonl_bytes"] += size
                selected_counts[f"stratum::{key}::documents"] += 1
                selected_counts[f"stratum::{key}::jsonl_bytes"] += size
            handle.flush()
            os.fsync(handle.fileno())
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "source_receipt_sha256": source_receipt["receipt_sha256"],
            "input_counts": dict(sorted(input_counts.items())),
            "selection": {
                "method": "dynamic-equal-stratum-high-quality-bottom-hash-v1",
                "maximum_jsonl_bytes": maximum_utf8_bytes,
                "strata": len(heaps),
                "development_excluded": True,
            },
            "sample": {
                "path": SAMPLE_NAME,
                "documents": selected_counts["documents"],
                "bytes": sample.stat().st_size,
                "sha256": sha256_file(sample),
                "ordered_jsonl_sha256": output_digest.hexdigest(),
                "ordered_document_identities_sha256": ordered_identities.hexdigest(),
            },
            "selected_counts": dict(sorted(selected_counts.items())),
            "source_text_persisted_only_in_bounded_sample": True,
            "tokenizer_measurement_only": True,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        if payload["sample"]["bytes"] != selected_bytes:
            raise TransientTokenizerSampleError("sample byte accounting differs")
        payload["receipt_sha256"] = canonical_sha256(payload)
        (stage / "receipt.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-utf8-bytes", type=int, required=True)
    args = parser.parse_args()
    result = build_sample(
        sys.stdin,
        args.source_receipt,
        args.output_root,
        maximum_utf8_bytes=args.maximum_utf8_bytes,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

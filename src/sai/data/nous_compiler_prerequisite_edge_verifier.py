"""Verify repeated-evidence prerequisite edges through resumable Hermès calls."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from sai.data.compiler_prerequisite_edge_labeling import (
    RUBRIC_SHA256,
    build_messages,
    normalize_candidate,
    normalize_model_judgment,
)
from sai.data.nous_compiler_worker import execute_contract, run_shard
from sai.data.nous_label_worker import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    NousLabelWorkerError,
    _post_json,
    _post_json_sse,
)

RECEIPT_SCHEMA = "sai-nous-compiler-prerequisite-edge-verification-receipt-v1"
SUMMARY_SCHEMA = "sai-nous-compiler-prerequisite-edge-verification-summary-v1"
OUTPUT_SUFFIX = "prerequisite-edge-verification"
REASONING_EFFORT = "low"
DEFAULT_CONCURRENCY = 2


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load one exact prerequisite-edge proposal population."""

    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise NousLabelWorkerError("prerequisite edge population is unsafe")
    rows = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    rows.append(normalize_candidate(json.loads(line)))
                except (json.JSONDecodeError, RuntimeError) as error:
                    raise NousLabelWorkerError(
                        f"prerequisite edge row {line_number} differs"
                    ) from error
    except (OSError, UnicodeError) as error:
        raise NousLabelWorkerError("prerequisite edge population differs") from error
    identities = [row["candidate_identity_sha256"] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise NousLabelWorkerError(
            "prerequisite edge identities are empty or duplicated"
        )
    return rows


def execute_one(
    candidate: dict[str, Any],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    maximum_attempts: int,
    stream_transport: bool = False,
    request_function: Any = _post_json,
    sleep_function: Any = time.sleep,
) -> dict[str, Any]:
    """Execute one strict directional prerequisite verification."""

    return execute_contract(
        candidate,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        maximum_attempts=maximum_attempts,
        build_messages_function=build_messages,
        normalize_function=normalize_model_judgment,
        rubric_sha256=RUBRIC_SHA256,
        receipt_schema=RECEIPT_SCHEMA,
        maximum_completion_tokens=4_200,
        reasoning_effort=REASONING_EFFORT,
        evidence_container_name="source_document",
        request_function=request_function,
        sleep_function=sleep_function,
        stream_transport=stream_transport,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="NOUS_API_KEY")
    parser.add_argument("--logical-shards", type=int, default=64)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--maximum-attempts", type=int, default=5)
    parser.add_argument("--stream-transport", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise NousLabelWorkerError(f"{args.api_key_env} is required")
    execute_function = execute_one
    if args.stream_transport:

        def execute_function(
            candidate: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            return execute_one(
                candidate,
                stream_transport=True,
                request_function=_post_json_sse,
                **kwargs,
            )

    summary = run_shard(
        args.candidates,
        args.output_root,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        logical_shards=args.logical_shards,
        shard_index=args.shard_index,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        maximum_attempts=args.maximum_attempts,
        execute_function=execute_function,
        load_function=load_candidates,
        summary_schema=SUMMARY_SCHEMA,
        rubric_sha256=RUBRIC_SHA256,
        output_suffix=OUTPUT_SUFFIX,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

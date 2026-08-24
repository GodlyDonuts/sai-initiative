"""Run independent provider reviews without merging them into primary labels."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.data_compiler_labeling import (
    RUBRIC_SHA256,
    build_messages,
    normalize_model_judgment,
    repair_evidence_quotes,
)
from sai.data.nous_compiler_worker import execute_contract, run_shard_locked
from sai.data.nous_label_worker import (
    NousLabelWorkerError,
    _post_json,
    _post_json_sse,
)

GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
COHERE_OPENAI_BASE_URL = "https://api.cohere.com/compatibility/v1"
CEREBRAS_OPENAI_BASE_URL = "https://api.cerebras.ai/v1"
ALLOWED_PROVIDER_MODELS = {
    GOOGLE_OPENAI_BASE_URL: frozenset(
        {
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b-it",
        }
    ),
    GROQ_OPENAI_BASE_URL: frozenset({"openai/gpt-oss-120b", "qwen/qwen3.6-27b"}),
    COHERE_OPENAI_BASE_URL: frozenset(
        {"command-a-plus-05-2026", "command-a-reasoning-08-2025"}
    ),
    CEREBRAS_OPENAI_BASE_URL: frozenset({"gemma-4-31b", "gpt-oss-120b"}),
}
ALLOWED_MODELS = frozenset().union(*ALLOWED_PROVIDER_MODELS.values())
RECEIPT_SCHEMA = "sai-independent-data-compiler-review-receipt-v1"
SUMMARY_SCHEMA = "sai-independent-data-compiler-review-shard-summary-v1"


def execute_one(
    candidate: dict[str, Any],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    maximum_attempts: int,
    stream_transport: bool = False,
    request_function: Callable[..., tuple[dict[str, Any], int]] = _post_json,
    sleep_function: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Review one row with explicit independent-family custody."""

    if model not in ALLOWED_PROVIDER_MODELS.get(base_url, frozenset()):
        raise NousLabelWorkerError("independent review provider/model identity differs")
    return execute_contract(
        candidate,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        maximum_attempts=maximum_attempts,
        request_function=request_function,
        sleep_function=sleep_function,
        build_messages_function=build_messages,
        normalize_function=normalize_model_judgment,
        rubric_sha256=RUBRIC_SHA256,
        receipt_schema=RECEIPT_SCHEMA,
        maximum_completion_tokens=2400,
        reasoning_effort=None,
        evidence_container_name="document",
        evidence_repair_function=repair_evidence_quotes,
        stream_transport=stream_transport,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(ALLOWED_MODELS), required=True)
    parser.add_argument(
        "--base-url", choices=sorted(ALLOWED_PROVIDER_MODELS), required=True
    )
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--logical-shards", type=int, default=128)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
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

    summary = run_shard_locked(
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
        summary_schema=SUMMARY_SCHEMA,
        rubric_sha256=RUBRIC_SHA256,
        output_suffix="independent-review",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

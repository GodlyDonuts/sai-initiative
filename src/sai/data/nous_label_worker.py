"""Run resumable, rate-limited Nous judgments over one deterministic data shard."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import (
    PERSPECTIVES,
    RUBRIC_SHA256,
    AgentLabelingError,
    _atomic_create,
    build_messages,
    normalize_candidate,
    normalize_model_judgment,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-nous-agent-label-receipt-v1"
DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_MODEL = "stealth/ox-alpha"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class NousLabelWorkerError(RuntimeError):
    """A request, response, or shard execution is invalid."""


def _validate_endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port == 8645
        and parsed.path.rstrip("/") == "/v1"
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    ):
        return "http://127.0.0.1:8645/v1"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "inference-api.nousresearch.com"
        or parsed.port not in (None, 443)
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise NousLabelWorkerError("Nous endpoint differs")
    return "https://inference-api.nousresearch.com/v1"


def _json_object(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise NousLabelWorkerError("model content is empty")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise NousLabelWorkerError("model content is not one JSON object") from error
    if not isinstance(payload, dict):
        raise NousLabelWorkerError("model content is not one JSON object")
    return payload


def _request_body(candidate: dict[str, Any], slot: int, model: str) -> dict[str, Any]:
    if not isinstance(model, str) or not model:
        raise NousLabelWorkerError("model differs")
    return {
        "model": model,
        "messages": build_messages(candidate, slot),
        "temperature": 0,
        "max_tokens": 1200,
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def _post_json(
    *,
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any], int]:
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=encoded,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "sai-data-labeler/1",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_bytes = response.read(4 << 20)
        if response.read(1):
            raise NousLabelWorkerError("model response exceeds size bound")
        status = response.status
    try:
        payload = json.loads(response_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NousLabelWorkerError("model response JSON differs") from error
    if not isinstance(payload, dict):
        raise NousLabelWorkerError("model response JSON differs")
    return payload, status


def execute_one(
    candidate: dict[str, Any],
    slot: int,
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    maximum_attempts: int,
    request_function: Callable[..., tuple[dict[str, Any], int]] = _post_json,
    sleep_function: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute one blinded judgment and retain exact request/response lineage."""

    candidate = normalize_candidate(candidate)
    base_url = _validate_endpoint(base_url)
    body = _request_body(candidate, slot, model)
    request_sha256 = canonical_sha256(body)
    attempts = []
    response: dict[str, Any] | None = None
    raw_judgment: dict[str, Any] | None = None
    judgment: dict[str, Any] | None = None
    choice: dict[str, Any] | None = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            response, status = request_function(
                base_url=base_url,
                api_key=api_key,
                body=body,
                timeout_seconds=timeout_seconds,
            )
            if status != 200:
                raise NousLabelWorkerError("successful request returned non-200")
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise NousLabelWorkerError("model response choices differ")
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(
                choice.get("message"), dict
            ):
                raise NousLabelWorkerError("model response message differs")
            raw_judgment = _json_object(choice["message"].get("content"))
            try:
                judgment = normalize_model_judgment(raw_judgment, candidate, slot)
            except AgentLabelingError as error:
                raise NousLabelWorkerError(
                    "model judgment violates the rubric"
                ) from error
            attempts.append(
                {"attempt": attempt, "http_status": status, "outcome": "valid"}
            )
            break
        except urllib.error.HTTPError as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": error.code,
                    "outcome": "transient_http_error",
                }
            )
            if error.code not in RETRYABLE_STATUS or attempt == maximum_attempts:
                raise NousLabelWorkerError(
                    f"Nous request failed with HTTP {error.code}"
                ) from error
        except (TimeoutError, urllib.error.URLError) as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": None,
                    "outcome": "transient_transport_error",
                }
            )
            if attempt == maximum_attempts:
                raise NousLabelWorkerError(
                    "Nous request exhausted transient retries"
                ) from error
        except NousLabelWorkerError:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": 200,
                    "outcome": "invalid_model_output",
                }
            )
            if attempt == maximum_attempts:
                raise
        delay = min(30.0, float(2 ** (attempt - 1)))
        sleep_function(delay)
    if response is None or raw_judgment is None or judgment is None or choice is None:
        raise NousLabelWorkerError("Nous request produced no response")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    usage_receipt = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        usage_receipt[field] = (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )
    response_identity = {
        "id": response.get("id") if isinstance(response.get("id"), str) else None,
        "model": (
            response.get("model") if isinstance(response.get("model"), str) else None
        ),
        "provider": (
            response.get("provider")
            if isinstance(response.get("provider"), str)
            else None
        ),
        "created": (
            response.get("created")
            if isinstance(response.get("created"), int)
            else None
        ),
        "finish_reason": (
            choice.get("finish_reason")
            if isinstance(choice.get("finish_reason"), str)
            else None
        ),
    }
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "annotator_slot": slot,
        "perspective": PERSPECTIVES[slot],
        "rubric_sha256": RUBRIC_SHA256,
        "endpoint_origin": base_url.rstrip("/"),
        "credential_transport": (
            "hermes_loopback_proxy"
            if base_url == "http://127.0.0.1:8645/v1"
            else "direct_portal_bearer"
        ),
        "requested_model": model,
        "request_sha256": request_sha256,
        "attempts": attempts,
        "response_identity": response_identity,
        "usage": usage_receipt,
        "raw_model_json_sha256": canonical_sha256(raw_judgment),
        "judgment": judgment,
        "api_key_persisted": False,
        "tools_enabled": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise NousLabelWorkerError("candidate population is missing or unsafe")
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(normalize_candidate(json.loads(line)))
            except (json.JSONDecodeError, AgentLabelingError) as error:
                raise NousLabelWorkerError(
                    f"candidate population row {line_number} differs"
                ) from error
    identities = [row["candidate_identity_sha256"] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise NousLabelWorkerError("candidate population is empty or duplicated")
    return rows


def _assigned(identity: str, logical_shards: int, shard_index: int) -> bool:
    return int(identity, 16) % logical_shards == shard_index


def run_shard(
    candidates_path: Path,
    output_root: Path,
    *,
    model: str,
    base_url: str,
    api_key: str,
    logical_shards: int,
    shard_index: int,
    concurrency: int,
    timeout_seconds: float,
    maximum_attempts: int,
    execute_function: Callable[..., dict[str, Any]] = execute_one,
) -> dict[str, Any]:
    """Run one deterministic logical shard with create-only row receipts."""

    if (
        isinstance(logical_shards, bool)
        or not 1 <= logical_shards <= 10_000
        or isinstance(shard_index, bool)
        or not 0 <= shard_index < logical_shards
        or isinstance(concurrency, bool)
        or not 1 <= concurrency <= 64
        or not api_key
    ):
        raise NousLabelWorkerError("worker geometry or credential differs")
    candidates = [
        row
        for row in _load_jsonl(candidates_path)
        if _assigned(row["candidate_identity_sha256"], logical_shards, shard_index)
    ]
    base_url = _validate_endpoint(base_url)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [(row, slot) for row in candidates for slot in range(len(PERSPECTIVES))]
    pending = []
    skipped = 0
    for row, slot in tasks:
        target = output_root / f"{row['candidate_identity_sha256']}.slot{slot}.json"
        if target.exists():
            skipped += 1
        else:
            pending.append((row, slot, target))

    def work(item: tuple[dict[str, Any], int, Path]) -> str:
        row, slot, target = item
        receipt = execute_function(
            row,
            slot,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            maximum_attempts=maximum_attempts,
        )
        _atomic_create(target, receipt)
        return receipt["receipt_sha256"]

    hashes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            hashes.append(future.result())
    summary = {
        "schema": "sai-nous-agent-label-shard-summary-v1",
        "status": "complete",
        "model": model,
        "rubric_sha256": RUBRIC_SHA256,
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "candidate_rows": len(candidates),
        "expected_judgments": len(tasks),
        "created_judgments": len(hashes),
        "preexisting_judgments": skipped,
        "created_receipts_sha256": canonical_sha256(sorted(hashes)),
        "api_key_persisted": False,
        "training_ready": False,
    }
    summary["receipt_sha256"] = canonical_sha256(summary)
    _atomic_create(output_root / f"shard_{shard_index:05d}.summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="NOUS_API_KEY")
    parser.add_argument("--logical-shards", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--maximum-attempts", type=int, default=5)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise NousLabelWorkerError(f"{args.api_key_env} is required")
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
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

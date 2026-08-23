"""Run persistent Nous/Hermes data-compiler judgments over exact source rows."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import tempfile
import time
import urllib.error
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_compiler_labeling import (
    REPRESENTATIONS,
    RISK_KEYS,
    RUBRIC_SHA256,
    build_messages,
    normalize_model_judgment,
    repair_evidence_quotes,
)
from sai.data.nous_label_worker import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    NousLabelWorkerError,
    _assigned,
    _json_object,
    _load_jsonl,
    _post_json,
    _post_json_sse,
    _retryable_http_status,
    _validate_endpoint,
)
from sai.data.token_stream import canonical_sha256

RECEIPT_SCHEMA = "sai-nous-data-compiler-receipt-v2"
SUMMARY_SCHEMA = "sai-nous-data-compiler-shard-summary-v2"
COMPILER_REASONING_EFFORT = "low"
DEFAULT_COMPILER_CONCURRENCY = 4
MAXIMUM_RESUME_RECEIPT_BYTES = 4 << 20
DEFAULT_SHARED_PROVIDER_CONCURRENCY = 16
HERMES_LOOPBACK_URL = "http://127.0.0.1:8645/v1"


def _shared_provider_concurrency(base_url: str) -> int | None:
    """Return the process-shared request ceiling for the local Hermes gateway."""

    if base_url != HERMES_LOOPBACK_URL:
        return None
    raw = os.environ.get(
        "SAI_NOUS_SHARED_PROVIDER_CONCURRENCY",
        str(DEFAULT_SHARED_PROVIDER_CONCURRENCY),
    )
    try:
        value = int(raw)
    except ValueError as error:
        raise NousLabelWorkerError("shared provider concurrency differs") from error
    if not 1 <= value <= 64 or str(value) != raw:
        raise NousLabelWorkerError("shared provider concurrency differs")
    return value


def _shared_provider_slot_root() -> Path:
    default = Path(tempfile.gettempdir()) / (
        f"sai-nous-provider-slots-{os.getuid()}-v1"
    )
    root = Path(os.environ.get("SAI_NOUS_SHARED_PROVIDER_SLOT_ROOT", str(default)))
    if not root.is_absolute() or root == Path(root.anchor):
        raise NousLabelWorkerError("shared provider slot root differs")
    try:
        root.mkdir(mode=0o700, exist_ok=True)
        metadata = root.lstat()
    except OSError as error:
        raise NousLabelWorkerError(
            "shared provider slot root is unavailable"
        ) from error
    if root.is_symlink() or not root.is_dir() or metadata.st_uid != os.getuid():
        raise NousLabelWorkerError("shared provider slot root is unsafe")
    return root


@contextmanager
def _shared_provider_request_slot(
    base_url: str,
    *,
    sleep_function: Callable[[float], None] = time.sleep,
) -> Iterator[int | None]:
    """Bound accepted provider pressure across independent worker processes."""

    concurrency = _shared_provider_concurrency(base_url)
    if concurrency is None:
        yield None
        return
    root = _shared_provider_slot_root()
    while True:
        for slot_index in range(concurrency):
            path = root / f"slot_{slot_index:03d}.lock"
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags, 0o600)
            except OSError as error:
                raise NousLabelWorkerError(
                    "shared provider slot cannot be opened"
                ) from error
            handle = os.fdopen(descriptor, "a+b")
            if os.fstat(handle.fileno()).st_uid != os.getuid():
                handle.close()
                raise NousLabelWorkerError("shared provider slot is unsafe")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield slot_index
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            return
        sleep_function(0.05)


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
    """Compile one source into a source-bound transformation plan."""

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
        reasoning_effort=COMPILER_REASONING_EFFORT,
        evidence_container_name="document",
        evidence_repair_function=repair_evidence_quotes,
        stream_transport=stream_transport,
    )


def execute_contract(
    candidate: dict[str, Any],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    maximum_attempts: int,
    build_messages_function: Callable[[dict[str, Any]], list[dict[str, str]]],
    normalize_function: Callable[[Any, dict[str, Any]], dict[str, Any]],
    rubric_sha256: str,
    receipt_schema: str,
    maximum_completion_tokens: int,
    reasoning_effort: str | None = None,
    evidence_container_name: str = "document",
    evidence_repair_function: (
        Callable[[Any, dict[str, Any]], tuple[Any, list[dict[str, Any]]]] | None
    ) = None,
    request_function: Callable[..., tuple[dict[str, Any], int]] = _post_json,
    sleep_function: Callable[[float], None] = time.sleep,
    stream_transport: bool = False,
) -> dict[str, Any]:
    """Execute one strict compiler contract without weakening its schema."""

    if (
        isinstance(maximum_completion_tokens, bool)
        or not 1 <= maximum_completion_tokens <= 16_384
        or not isinstance(receipt_schema, str)
        or not receipt_schema
        or not isinstance(rubric_sha256, str)
        or len(rubric_sha256) != 64
        or any(character not in "0123456789abcdef" for character in rubric_sha256)
        or reasoning_effort not in {None, "none", "minimal", "low", "medium", "high"}
        or evidence_container_name
        not in {"document", "book_excerpt", "source_document"}
        or not isinstance(stream_transport, bool)
    ):
        raise NousLabelWorkerError("compiler contract identity or token bound differs")
    base_url = _validate_endpoint(base_url)
    body = {
        "model": model,
        "messages": build_messages_function(candidate),
        "temperature": 0,
        "max_tokens": maximum_completion_tokens,
        "stream": stream_transport,
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    request_sha256 = canonical_sha256(body)
    base_messages = list(body["messages"])
    attempts = []
    attempt_request_sha256s = []
    response = None
    raw_judgment = None
    judgment = None
    evidence_repairs: list[dict[str, Any]] = []
    choice = None
    shared_provider_concurrency = _shared_provider_concurrency(base_url)
    for attempt in range(1, maximum_attempts + 1):
        attempt_request_sha256 = canonical_sha256(body)
        attempt_request_sha256s.append(attempt_request_sha256)
        try:
            with _shared_provider_request_slot(base_url, sleep_function=sleep_function):
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
            normalized_payload = raw_judgment
            evidence_repairs = []
            if evidence_repair_function is not None:
                normalized_payload, evidence_repairs = evidence_repair_function(
                    raw_judgment, candidate
                )
            judgment = normalize_function(normalized_payload, candidate)
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": status,
                    "outcome": "valid",
                    "request_sha256": attempt_request_sha256,
                }
            )
            break
        except urllib.error.HTTPError as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": error.code,
                    "outcome": "transient_http_error",
                    "request_sha256": attempt_request_sha256,
                }
            )
            if (
                not _retryable_http_status(error.code, base_url)
                or attempt == maximum_attempts
            ):
                raise NousLabelWorkerError(
                    f"Nous request failed with HTTP {error.code}"
                ) from error
        except (TimeoutError, urllib.error.URLError) as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": None,
                    "outcome": "transient_transport_error",
                    "request_sha256": attempt_request_sha256,
                }
            )
            if attempt == maximum_attempts:
                raise NousLabelWorkerError(
                    "Nous request exhausted transient retries"
                ) from error
        except (NousLabelWorkerError, RuntimeError) as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": 200,
                    "outcome": "invalid_model_output",
                    "request_sha256": attempt_request_sha256,
                }
            )
            if attempt == maximum_attempts:
                raise
            prior_content = None
            if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                value = choice["message"].get("content")
                if isinstance(value, str) and 0 < len(value) <= (1 << 20):
                    prior_content = value
            if prior_content is not None:
                validation_hint = ""
                if "concepts differs" in str(error):
                    validation_hint = (
                        " concepts_taught must be a JSON list containing at most "
                        "20 unique, nonempty, lowercase strings; each string must "
                        "be at most 96 characters. Do not repeat a concept and do "
                        "not use title case, symbols as standalone entries, or "
                        "nested objects."
                    )
                elif (
                    evidence_container_name == "document"
                    and "risks fields differ" in str(error)
                ):
                    validation_hint = (
                        " risks must be a JSON object with exactly these keys, "
                        "each mapped to true or false: "
                        + ", ".join(RISK_KEYS)
                        + ". Do not omit, rename, or add any risk key."
                    )
                elif (
                    evidence_container_name == "document"
                    and "recommended representations differs" in str(error)
                ):
                    validation_hint = (
                        " recommended_representations must be a JSON list of 1..8 "
                        "unique strings chosen only from: "
                        + ", ".join(REPRESENTATIONS)
                        + ". Do not invent, repeat, or combine labels."
                    )
                elif (
                    evidence_container_name == "document"
                    and "non-English translation plan differs" in str(error)
                ):
                    validation_hint = (
                        " source_language means the predominant language of the "
                        "actual supplied document, not a language, title, author, "
                        "or work merely discussed inside it. If the document is "
                        "English, set source_language=english, "
                        "translation_disposition=not_needed_english, and "
                        "translation_priority=0. If a retained document is "
                        "non-English, include english_translation in "
                        "recommended_representations, use a non-English translation "
                        "disposition, and set translation_priority to 1..4."
                        " A disconnected catalog form, field list, or metadata record "
                        "that lacks coherent educational or expressive content should "
                        "be rejected with curriculum_phase=reject and "
                        "preservation_policy=reject; do not retain it merely because "
                        "it names a valuable work."
                    )
                body["messages"] = [
                    *base_messages,
                    {"role": "assistant", "content": prior_content},
                    {
                        "role": "user",
                        "content": (
                            "Your JSON failed strict validation: "
                            f"{str(error)[:256]}. Return a corrected complete JSON "
                            "object with the exact same required keys. Do not defend "
                            "the prior answer. Remove any claim or edge that cannot be "
                            "supported by a byte-for-byte quote from "
                            f"{evidence_container_name}."
                            + validation_hint
                            + (
                                " You MUST replace evidence_quotes only with one "
                                "to four complete, exact strings copied from "
                                "evidence_quote_candidates. Do not shorten, "
                                "normalize, join, or rewrite those strings."
                                if evidence_container_name == "document"
                                else ""
                            )
                        ),
                    },
                ]
        sleep_function(min(30.0, float(2 ** (attempt - 1))))
    if response is None or judgment is None or choice is None or raw_judgment is None:
        raise NousLabelWorkerError("Nous request produced no compiler response")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    usage_receipt = {
        field: (
            usage.get(field)
            if isinstance(usage.get(field), int)
            and not isinstance(usage.get(field), bool)
            and usage[field] >= 0
            else None
        )
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    receipt = {
        "schema": receipt_schema,
        "status": "complete",
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "rubric_sha256": rubric_sha256,
        "endpoint_origin": base_url.rstrip("/"),
        "credential_transport": (
            "hermes_loopback_proxy"
            if base_url == HERMES_LOOPBACK_URL
            else "direct_portal_bearer"
        ),
        "shared_provider_concurrency_limit": shared_provider_concurrency,
        "requested_model": model,
        "request_sha256": request_sha256,
        "attempt_request_sha256s": attempt_request_sha256s,
        "successful_request_sha256": attempt_request_sha256s[-1],
        "request_reasoning_effort": reasoning_effort,
        "request_stream_transport": stream_transport,
        "response_stream_transport": {
            "requested": stream_transport,
            "done_marker_observed": (
                response.get("_sse_done_marker_observed")
                if stream_transport
                and isinstance(response.get("_sse_done_marker_observed"), bool)
                else None
            ),
            "terminal_finish_reason_observed": isinstance(
                choice.get("finish_reason"), str
            ),
        },
        "attempts": attempts,
        "response_identity": {
            "id": response.get("id") if isinstance(response.get("id"), str) else None,
            "model": (
                response.get("model")
                if isinstance(response.get("model"), str)
                else None
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
        },
        "usage": usage_receipt,
        "raw_model_json_sha256": canonical_sha256(raw_judgment),
        "deterministic_evidence_quote_repairs": evidence_repairs,
        "judgment": judgment,
        "api_key_persisted": False,
        "tools_enabled": False,
        "raw_source_is_training_data": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _resume_completed_shard(
    summary_path: Path,
    output_root: Path,
    candidates: list[dict[str, Any]],
    *,
    model: str,
    logical_shards: int,
    shard_index: int,
    summary_schema: str,
    rubric_sha256: str,
    output_suffix: str,
) -> dict[str, Any] | None:
    """Replay a complete shard only after every exact receipt is revalidated."""

    if not summary_path.exists() and not summary_path.is_symlink():
        return None
    if (
        not summary_path.is_file()
        or summary_path.is_symlink()
        or summary_path.stat().st_nlink != 1
        or summary_path.stat().st_size > MAXIMUM_RESUME_RECEIPT_BYTES
    ):
        raise NousLabelWorkerError("completed compiler shard summary is unsafe")
    try:
        summary = json.loads(summary_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise NousLabelWorkerError(
            "completed compiler shard summary cannot be decoded"
        ) from error
    if not isinstance(summary, dict):
        raise NousLabelWorkerError("completed compiler shard summary differs")
    unsigned = {key: value for key, value in summary.items() if key != "receipt_sha256"}
    if (
        summary.get("receipt_sha256") != canonical_sha256(unsigned)
        or summary.get("schema") != summary_schema
        or summary.get("status") != "complete"
        or summary.get("model") != model
        or summary.get("rubric_sha256") != rubric_sha256
        or summary.get("logical_shards") != logical_shards
        or summary.get("shard_index") != shard_index
        or summary.get("candidate_rows") != len(candidates)
        or summary.get("expected_judgments") != len(candidates)
        or not isinstance(summary.get("created_judgments"), int)
        or not isinstance(summary.get("preexisting_judgments"), int)
        or summary["created_judgments"] + summary["preexisting_judgments"]
        != len(candidates)
        or summary.get("api_key_persisted") is not False
        or summary.get("training_ready") is not False
    ):
        raise NousLabelWorkerError("completed compiler shard summary differs")
    for row in candidates:
        identity = row["candidate_identity_sha256"]
        target = output_root / f"{identity}.{output_suffix}.json"
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_nlink != 1
            or target.stat().st_size > MAXIMUM_RESUME_RECEIPT_BYTES
        ):
            raise NousLabelWorkerError(
                "completed compiler shard has missing or unsafe receipt"
            )
        try:
            receipt = json.loads(target.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise NousLabelWorkerError(
                "completed compiler shard receipt cannot be decoded"
            ) from error
        if not isinstance(receipt, dict):
            raise NousLabelWorkerError("completed compiler shard receipt differs")
        receipt_unsigned = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        if (
            receipt.get("candidate_identity_sha256") != identity
            or receipt.get("receipt_sha256") != canonical_sha256(receipt_unsigned)
            or receipt.get("api_key_persisted", False) is not False
            or receipt.get("training_ready", False) is not False
        ):
            raise NousLabelWorkerError("completed compiler shard receipt differs")
    return summary


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
    load_function: Callable[[Path], list[dict[str, Any]]] = _load_jsonl,
    summary_schema: str = SUMMARY_SCHEMA,
    rubric_sha256: str = RUBRIC_SHA256,
    output_suffix: str = "compiler",
) -> dict[str, Any]:
    """Run one persistent compiler shard while isolating individual failures."""

    if (
        isinstance(logical_shards, bool)
        or not 1 <= logical_shards <= 10_000
        or isinstance(shard_index, bool)
        or not 0 <= shard_index < logical_shards
        or isinstance(concurrency, bool)
        or not 1 <= concurrency <= 64
        or not api_key
        or not isinstance(summary_schema, str)
        or not summary_schema
        or not isinstance(output_suffix, str)
        or not output_suffix
        or len(output_suffix) > 64
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in output_suffix
        )
        or not isinstance(rubric_sha256, str)
        or len(rubric_sha256) != 64
        or any(character not in "0123456789abcdef" for character in rubric_sha256)
    ):
        raise NousLabelWorkerError("compiler worker geometry or credential differs")
    candidates = [
        row
        for row in load_function(candidates_path)
        if _assigned(row["candidate_identity_sha256"], logical_shards, shard_index)
    ]
    base_url = _validate_endpoint(base_url)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"shard_{shard_index:05d}.summary.json"
    completed = _resume_completed_shard(
        summary_path,
        output_root,
        candidates,
        model=model,
        logical_shards=logical_shards,
        shard_index=shard_index,
        summary_schema=summary_schema,
        rubric_sha256=rubric_sha256,
        output_suffix=output_suffix,
    )
    if completed is not None:
        return completed
    pending = []
    skipped = 0
    for row in candidates:
        target = output_root / (
            f"{row['candidate_identity_sha256']}.{output_suffix}.json"
        )
        if target.exists():
            skipped += 1
        else:
            pending.append((row, target))

    def work(item: tuple[dict[str, Any], Path]) -> str:
        row, target = item
        receipt = execute_function(
            row,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            maximum_attempts=maximum_attempts,
        )
        _atomic_create(target, receipt)
        return receipt["receipt_sha256"]

    hashes = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(work, item): item for item in pending}
        for future in concurrent.futures.as_completed(futures):
            row, _target = futures[future]
            try:
                hashes.append(future.result())
            except Exception as error:  # noqa: BLE001 - isolate remote row failure
                failures.append(
                    {
                        "candidate_identity_sha256": row["candidate_identity_sha256"],
                        "error_type": type(error).__name__,
                        "error": str(error)[:512],
                    }
                )
            completed = len(hashes) + len(failures)
            if completed % 100 == 0 or completed == len(pending):
                print(
                    json.dumps(
                        {
                            "event": "compiler_progress",
                            "shard_index": shard_index,
                            "created": len(hashes),
                            "failed": len(failures),
                            "pending": len(pending) - completed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if failures:
        raise NousLabelWorkerError(
            f"{len(failures)} compiler judgment(s) failed; receipts are resumable; "
            f"first={json.dumps(failures[0], sort_keys=True)}"
        )
    summary = {
        "schema": summary_schema,
        "status": "complete",
        "model": model,
        "rubric_sha256": rubric_sha256,
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "candidate_rows": len(candidates),
        "expected_judgments": len(candidates),
        "created_judgments": len(hashes),
        "preexisting_judgments": skipped,
        "created_receipts_sha256": canonical_sha256(sorted(hashes)),
        "api_key_persisted": False,
        "training_ready": False,
    }
    summary["receipt_sha256"] = canonical_sha256(summary)
    _atomic_create(summary_path, summary)
    return summary


def run_shard_locked(
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
    load_function: Callable[[Path], list[dict[str, Any]]] = _load_jsonl,
    summary_schema: str = SUMMARY_SCHEMA,
    rubric_sha256: str = RUBRIC_SHA256,
    output_suffix: str = "compiler",
) -> dict[str, Any]:
    """Serialize one logical shard across independent resumable workers."""

    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise NousLabelWorkerError("compiler shard lock identity differs")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / f".shard_{shard_index:05d}.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return run_shard(
            candidates_path,
            output_root,
            model=model,
            base_url=base_url,
            api_key=api_key,
            logical_shards=logical_shards,
            shard_index=shard_index,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            maximum_attempts=maximum_attempts,
            execute_function=execute_function,
            load_function=load_function,
            summary_schema=summary_schema,
            rubric_sha256=rubric_sha256,
            output_suffix=output_suffix,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="NOUS_API_KEY")
    parser.add_argument("--logical-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_COMPILER_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--maximum-attempts", type=int, default=5)
    parser.add_argument(
        "--stream-transport",
        action="store_true",
        help=(
            "Request SSE and reconstruct one final response to avoid idle HTTP "
            "timeouts."
        ),
    )
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
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

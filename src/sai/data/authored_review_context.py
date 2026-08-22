"""Preflight blinded curriculum-review prompts against one sealed tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.authored_review_model import (
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    REVIEWERS,
    _blind_inputs,
    _concept_prompt,
    _messages,
    _prompt,
)
from sai.data.external_hf_snapshot import validate_external_snapshot
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-authored-curriculum-review-context-preflight-v1"


class AuthoredReviewContextError(RuntimeError):
    """The reviewer context preflight or its immutable receipt differs."""


def _tokenizer(model_root: Path) -> tuple[Any, str]:
    try:
        import transformers
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=False
        )
    except Exception as error:
        raise AuthoredReviewContextError("reviewer tokenizer load differs") from error
    return tokenizer, transformers.__version__


def _payload(
    *,
    reviewer: str,
    model_root: Path,
    manifest: Path,
    restoration_receipt: Path,
    review_packet: Path,
    review_packet_receipt: Path,
    expected_review_packet_sha256: str,
    expected_review_packet_receipt_sha256: str,
    concept_list: Path,
    annotation_policy: Path,
) -> dict[str, Any]:
    if reviewer not in REVIEWERS:
        raise AuthoredReviewContextError("reviewer identity differs")
    try:
        snapshot = validate_external_snapshot(
            model_root,
            manifest_path=manifest,
            receipt_path=restoration_receipt,
            spec=REVIEWERS[reviewer],
        )
        inputs = _blind_inputs(
            review_packet=review_packet,
            review_packet_receipt=review_packet_receipt,
            expected_review_packet_sha256=expected_review_packet_sha256,
            expected_review_packet_receipt_sha256=(
                expected_review_packet_receipt_sha256
            ),
            concept_list=concept_list,
            annotation_policy=annotation_policy,
        )
    except Exception as error:
        raise AuthoredReviewContextError("reviewer inputs differ") from error
    tokenizer, transformers_version = _tokenizer(model_root)
    try:
        import torch
    except ImportError as error:
        raise AuthoredReviewContextError("review tokenizer runtime differs") from error
    concept_prompt = _concept_prompt(inputs.concept_payload)
    rows = []
    for index, source in enumerate(inputs.packet):
        prompt = _prompt(source, concept_prompt, None)
        try:
            encoded = tokenizer.apply_chat_template(
                _messages(prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except Exception as error:
            raise AuthoredReviewContextError(
                "review prompt tokenization differs"
            ) from error
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.ndim != 2
            or encoded.shape[0] != 1
            or encoded.shape[1] <= 0
        ):
            raise AuthoredReviewContextError("review prompt tokenization differs")
        rows.append(
            {
                "index": index,
                "review_identity_sha256": source["review_identity_sha256"],
                "input_tokens": int(encoded.shape[1]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    counts = [row["input_tokens"] for row in rows]
    if len(rows) != 127 or max(counts) > MAX_INPUT_TOKENS:
        raise AuthoredReviewContextError("review prompt exceeds frozen context budget")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "reviewer": reviewer,
        "snapshot": snapshot,
        "transformers_version": transformers_version,
        "tokenizer_length": len(tokenizer),
        "blind_review_packet_sha256": hashlib.sha256(inputs.packet_encoded).hexdigest(),
        "blind_review_packet_receipt_sha256": hashlib.sha256(
            inputs.packet_receipt_encoded
        ).hexdigest(),
        "concept_list_sha256": hashlib.sha256(inputs.concept_encoded).hexdigest(),
        "annotation_policy_sha256": hashlib.sha256(inputs.policy_encoded).hexdigest(),
        "row_count": len(rows),
        "minimum_input_tokens": min(counts),
        "maximum_input_tokens_observed": max(counts),
        "maximum_input_tokens_allowed": MAX_INPUT_TOKENS,
        "maximum_new_tokens": MAX_NEW_TOKENS,
        "rows": rows,
        "gpu_jobs_submitted": 0,
        "model_loaded": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def run(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    payload = _payload(**kwargs)
    _write_create_only(
        output, json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    )
    return payload


def validate(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    try:
        encoded = _read_regular_bytes(output, maximum_bytes=4 << 20)
        actual = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredReviewContextError("context receipt differs") from error
    expected = _payload(**kwargs)
    if actual != expected:
        raise AuthoredReviewContextError("context receipt differs")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--reviewer", choices=sorted(REVIEWERS), required=True)
    for name in (
        "model-root",
        "manifest",
        "restoration-receipt",
        "review-packet",
        "review-packet-receipt",
        "concept-list",
        "annotation-policy",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-review-packet-sha256", required=True)
    parser.add_argument("--expected-review-packet-receipt-sha256", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    payload = (run if command == "generate" else validate)(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "reviewer": payload["reviewer"],
                "rows": payload["row_count"],
                "maximum_input_tokens_observed": payload[
                    "maximum_input_tokens_observed"
                ],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

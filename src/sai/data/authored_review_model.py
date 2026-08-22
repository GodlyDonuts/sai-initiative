"""Generate candidate semantic-review drafts with one sealed offline HF model.

The reviewer receives only the immutable blind packet, concept list, and
annotation policy.  It never receives the hidden chapter-order key.  Generated
labels must already satisfy the exact-quote compiler before they are preserved.
Model review is triage evidence only and never substitutes for independent
human review, data admission, training authorization, or a scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.annotation_policy import AnnotationPolicyError, validate_policy
from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.authored_review_compile import (
    DRAFT_SCHEMA,
    AuthoredReviewCompileError,
    _compile_rows,
)
from sai.data.authored_review_packet import REVIEW_SCHEMA
from sai.data.external_hf_snapshot import (
    ExternalSnapshotError,
    ExternalSnapshotSpec,
    sha256_file,
    validate_external_snapshot,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-authored-curriculum-model-review-draft-receipt-v1"
RAW_SCHEMA = "sai-authored-curriculum-model-review-raw-row-v1"
FAILURE_SCHEMA = "sai-authored-curriculum-model-review-failure-v1"
PACKET_RECEIPT_SCHEMA = "sai-authored-curriculum-review-packet-receipt-v1"
MAX_INPUT_TOKENS = 24_576
MAX_NEW_TOKENS = 2_048
MAX_ATTEMPTS = 3
MAX_ASSUMED_CONCEPTS = 12
MAX_TAUGHT_CONCEPTS = 8
MAX_EVIDENCE_QUOTES_PER_CONCEPT = 2
MAX_DEFECTS = 4
REVIEWERS = {
    "qwen35_9b": ExternalSnapshotSpec(
        repository="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        tree_sha256="37048cc496c8992ea778fc1395f10b3c1d2dcb434f5de066f9f5c4bbf832903a",
        manifest_sha256="cd52186b26f92a30d68e0449826a6bb079b30633f3c4142f10df0ecc0683bd81",
        receipt_sha256="a72f6325de1f09c08d92498eb9367f720dce571bd69ba298e655590d68c0a123",
    ),
    "smollm3_3b": ExternalSnapshotSpec(
        repository="HuggingFaceTB/SmolLM3-3B",
        revision="a07cc9a04f16550a088caea529712d1d335b0ac1",
        tree_sha256="6badcd593aee3052e3d66afb315b979e2cc62c4a61f9cef31c07203912478a0f",
        manifest_sha256="e689bcce197b02c4d2e8b600696ec3137b1e1724104954cc1735d5d8848e6945",
        receipt_sha256="4672fc549809d89f0489a5e82045d54d3b5580718dcf40631a31807fd7415c85",
    ),
}
_PACKET_KEYS = {
    "schema",
    "review_identity_sha256",
    "title",
    "text_sha256",
    "text",
    "requested_review",
}
_REQUEST = {
    "instructional_quality": "unlabeled",
    "assumed_prior_concepts": [],
    "taught_concepts_with_evidence_spans": [],
    "extraction_or_factual_defects": [],
    "admission_recommendation": "unlabeled",
}
_RESPONSE_KEYS = {
    "instructional_quality_ppm",
    "assumed_prior_concepts",
    "taught_concepts",
    "defects",
    "admission_recommendation",
}
_RAW_KEYS = {
    "schema",
    "index",
    "review_identity_sha256",
    "source_text_sha256",
    "attempts",
    "draft",
}
_ATTEMPT_KEYS = {
    "attempt",
    "prompt_sha256",
    "input_tokens",
    "output_tokens",
    "response",
    "response_sha256",
    "accepted",
    "rejection",
}
_FAILURE_KEYS = {
    "schema",
    "status",
    "index",
    "review_identity_sha256",
    "source_text_sha256",
    "attempts",
    "human_review_completed",
    "audit_qualified",
    "training_authorized",
    "four_b_training_authorized",
    "receipt_sha256",
}
_IDENTITY_KEYS = {
    "schema",
    "status",
    "reviewer",
    "runtime",
    "blind_review_packet_sha256",
    "blind_review_packet_receipt_sha256",
    "concept_list_sha256",
    "annotation_policy_sha256",
    "runner_source_sha256",
    "generation",
    "hidden_review_key_accessed",
    "human_review_completed",
    "audit_qualified",
    "training_authorized",
    "four_b_training_authorized",
    "receipt_sha256",
}
_RUNTIME_KEYS = {
    "reviewer",
    "snapshot",
    "model_class",
    "parameter_count",
    "maximum_position_embeddings",
    "tokenizer_length",
    "unexpected_weight_count",
    "unexpected_weights_sha256",
    "transformers_version",
    "model_source_sha256",
    "python",
    "torch",
    "cuda",
    "gpu_name",
    "gpu_capability",
}


class AuthoredModelReviewError(RuntimeError):
    """The blinded model-review boundary or generated evidence differs."""


@dataclass(frozen=True)
class _Inputs:
    packet: list[dict[str, Any]]
    packet_encoded: bytes
    packet_receipt_encoded: bytes
    concept_payload: dict[str, Any]
    concept_encoded: bytes
    policy_encoded: bytes
    policy: dict[str, Any]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, label: str, maximum: int = 8 << 20) -> tuple[Any, bytes]:
    try:
        encoded = _read_regular_bytes(path, maximum_bytes=maximum)
        return json.loads(encoded), encoded
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise AuthoredModelReviewError(f"{label} differs") from error


def _jsonl(path: Path, label: str, maximum: int = 1 << 30) -> tuple[list[Any], bytes]:
    try:
        encoded = _read_regular_bytes(path, maximum_bytes=maximum)
        rows = [json.loads(line) for line in encoded.decode().splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise AuthoredModelReviewError(f"{label} differs") from error
    if not rows:
        raise AuthoredModelReviewError(f"{label} differs")
    return rows, encoded


def _blind_inputs(
    *,
    review_packet: Path,
    review_packet_receipt: Path,
    expected_review_packet_sha256: str,
    expected_review_packet_receipt_sha256: str,
    concept_list: Path,
    annotation_policy: Path,
) -> _Inputs:
    packet_rows, packet_encoded = _jsonl(review_packet, "blind review packet")
    receipt, receipt_encoded = _json(
        review_packet_receipt, "blind review receipt", 1 << 20
    )
    concept_payload, concept_encoded = _json(concept_list, "concept list")
    try:
        policy_encoded = _read_regular_bytes(annotation_policy, maximum_bytes=1 << 20)
        policy = validate_policy(
            annotation_policy,
            expected_concept_list_sha256=_sha256(concept_encoded),
        )
    except (AnnotationPolicyError, OSError) as error:
        raise AuthoredModelReviewError("annotation policy differs") from error
    if (
        expected_review_packet_sha256 != _sha256(packet_encoded)
        or expected_review_packet_receipt_sha256 != _sha256(receipt_encoded)
        or not isinstance(receipt, dict)
        or receipt.get("schema") != PACKET_RECEIPT_SCHEMA
        or receipt.get("status") != "awaiting_independent_review"
        or receipt.get("training_authorized") is not False
        or receipt.get("four_b_training_authorized") is not False
        or receipt.get("review_output", {}).get("sha256") != _sha256(packet_encoded)
        or receipt.get("review_output", {}).get("bytes") != len(packet_encoded)
        or receipt.get("review_output", {}).get("rows") != len(packet_rows)
        or len(packet_rows) != 127
    ):
        raise AuthoredModelReviewError("blind review receipt differs")
    identities = []
    for row in packet_rows:
        if (
            not isinstance(row, dict)
            or set(row) != _PACKET_KEYS
            or row["schema"] != REVIEW_SCHEMA
            or row["requested_review"] != _REQUEST
            or not isinstance(row["review_identity_sha256"], str)
            or len(row["review_identity_sha256"]) != 64
            or not isinstance(row["title"], str)
            or not row["title"]
            or not isinstance(row["text"], str)
            or not row["text"]
            or row["text_sha256"] != _sha256(row["text"].encode())
        ):
            raise AuthoredModelReviewError("blind review row differs")
        identities.append(row["review_identity_sha256"])
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise AuthoredModelReviewError("blind review ordering differs")
    if (
        not isinstance(concept_payload, dict)
        or concept_payload.get("schema") != "sai-semantic-prerequisite-concept-list-v1"
        or concept_payload.get("status") != "candidate"
        or not isinstance(concept_payload.get("concepts"), list)
        or not concept_payload["concepts"]
    ):
        raise AuthoredModelReviewError("concept list differs")
    return _Inputs(
        packet=packet_rows,
        packet_encoded=packet_encoded,
        packet_receipt_encoded=receipt_encoded,
        concept_payload=concept_payload,
        concept_encoded=concept_encoded,
        policy_encoded=policy_encoded,
        policy=policy,
    )


def _concept_prompt(concept_payload: dict[str, Any]) -> str:
    lines = []
    for concept in concept_payload["concepts"]:
        if not isinstance(concept, dict):
            raise AuthoredModelReviewError("concept list differs")
        concept_id = concept.get("concept_id")
        name = concept.get("name")
        if not isinstance(concept_id, str) or not isinstance(name, str):
            raise AuthoredModelReviewError("concept list differs")
        lines.append(f"- {concept_id}: {name}")
    return "\n".join(lines)


def _prompt(source: dict[str, Any], concept_prompt: str, repair: str | None) -> str:
    repair_text = (
        "\nYour prior response was rejected for this exact reason: "
        + repair
        + "\nDiscard that response and return a smaller corrected object from the "
        "original chapter. If a quote or evidence label was rejected, remove every "
        "label whose quote is not a literal chapter substring. Do not explain the "
        "correction."
        if repair is not None
        else ""
    )
    schema = {
        "instructional_quality_ppm": "integer",
        "assumed_prior_concepts": ["sorted concept ids"],
        "taught_concepts": [
            {
                "concept_id": "id",
                "confidence_ppm": "integer",
                "evidence_quotes": ["exact quote"],
            }
        ],
        "defects": [{"category": "category", "evidence_quote": "exact quote"}],
        "admission_recommendation": "admit|exclude|revise",
    }
    lines = [
        "You are performing a BLIND candidate semantic review of one "
        "instructional chapter.",
        "You do not know its source path, publisher order, stage, or declared "
        "prerequisites.",
        "Use only the chapter text and allowed concept list below.",
        "",
        "Policy:",
        "- A taught concept requires explicit instruction or demonstrated use and "
        "at least one EXACT VERBATIM quote copied from the chapter.",
        "- Every quote must contain at least 16 Unicode codepoints and occur only "
        "once in the chapter.",
        "- Evidence quotes may only be copied from between CHAPTER START and CHAPTER "
        "END. Never quote the title, policy, JSON shape, concept IDs, or allowed "
        "concept descriptions. Never paraphrase.",
        "- assumed_prior_concepts are concepts the chapter relies upon without "
        "teaching.",
        "- A concept cannot be both taught and assumed.",
        f"- Include at most {MAX_ASSUMED_CONCEPTS} assumed concepts and at most "
        f"{MAX_TAUGHT_CONCEPTS} taught concepts. Select the smallest direct set; "
        "do not list every background concept that a reader might know.",
        "- Copy concept IDs exactly from the allowed list. Never invent an ID, "
        "change its namespace, repeat an ID, or repeat a list item.",
        "- Sort both assumed_prior_concepts and taught_concepts by exact concept_id.",
        f"- Use at most {MAX_EVIDENCE_QUOTES_PER_CONCEPT} nonoverlapping evidence "
        "quotes per taught concept and at most "
        f"{MAX_DEFECTS} defects.",
        "- confidence_ppm must be 800000..1000000.",
        "- instructional_quality_ppm must be 0..1000000.",
        "- If no concept is taught, recommendation cannot be admit.",
        "- If you cannot produce a supported taught label, use an empty taught list "
        "and recommend exclude or revise.",
        "- Defect category is one of extraction, factual, pedagogical, licensing "
        "and also requires one exact quote.",
        "- Do not infer facts from the title or from outside knowledge.",
        "",
        "Allowed concepts:",
        concept_prompt,
        "",
        "Return exactly one JSON object and no commentary using this shape. Sort "
        "taught_concepts by concept_id. End immediately after the final closing "
        "brace:",
        json.dumps(schema, sort_keys=True, separators=(",", ":")),
        "",
        f"Review identity: {source['review_identity_sha256']}",
        f"Title: {source['title']}",
        "CHAPTER START",
        source["text"],
        f"CHAPTER END{repair_text}",
    ]
    return "\n".join(lines)


def _response_object(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise AuthoredModelReviewError("model response is not exact JSON") from error
    if not isinstance(payload, dict) or set(payload) != _RESPONSE_KEYS:
        raise AuthoredModelReviewError("model response fields differ")
    return payload


def _canonical_quote(source_text: str, quote: str) -> str:
    if source_text.count(quote) == 1:
        return quote
    tokens = quote.split()
    if not tokens:
        return quote
    matches = list(
        re.finditer(r"\s+".join(re.escape(token) for token in tokens), source_text)
    )
    if len(matches) != 1:
        return quote
    return source_text[matches[0].start() : matches[0].end()]


def _draft_from_response(
    source: dict[str, Any],
    response: str,
    concept_ids: set[str],
    policy: dict[str, Any],
) -> dict[str, Any]:
    payload = _response_object(response)
    assumed = payload["assumed_prior_concepts"]
    taught = payload["taught_concepts"]
    defects = payload["defects"]
    if (
        not isinstance(assumed, list)
        or len(assumed) > MAX_ASSUMED_CONCEPTS
        or not isinstance(taught, list)
        or len(taught) > MAX_TAUGHT_CONCEPTS
        or not isinstance(defects, list)
        or len(defects) > MAX_DEFECTS
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("evidence_quotes"), list)
            or len(item["evidence_quotes"]) > MAX_EVIDENCE_QUOTES_PER_CONCEPT
            for item in taught
        )
    ):
        raise AuthoredModelReviewError("model response exceeds review evidence limits")
    if all(isinstance(value, str) for value in assumed):
        payload["assumed_prior_concepts"] = sorted(assumed)
    if all(
        isinstance(value, dict) and isinstance(value.get("concept_id"), str)
        for value in taught
    ):
        payload["taught_concepts"] = sorted(
            taught, key=lambda value: value["concept_id"]
        )
    for value in taught:
        if isinstance(value, dict) and isinstance(value.get("evidence_quotes"), list):
            value["evidence_quotes"] = [
                (
                    _canonical_quote(source["text"], quote)
                    if isinstance(quote, str)
                    else quote
                )
                for quote in value["evidence_quotes"]
            ]
    for value in defects:
        if isinstance(value, dict) and isinstance(value.get("evidence_quote"), str):
            value["evidence_quote"] = _canonical_quote(
                source["text"], value["evidence_quote"]
            )
    if all(isinstance(value, str) for value in assumed) and all(
        isinstance(value, dict) and isinstance(value.get("concept_id"), str)
        for value in taught
    ):
        taught_ids = {value["concept_id"] for value in taught}
        payload["assumed_prior_concepts"] = [
            concept_id
            for concept_id in payload["assumed_prior_concepts"]
            if concept_id not in taught_ids
        ]
        if not taught_ids and payload.get("admission_recommendation") == "admit":
            payload["admission_recommendation"] = "revise"
    draft = {
        "schema": DRAFT_SCHEMA,
        "review_identity_sha256": source["review_identity_sha256"],
        **payload,
    }
    try:
        _compile_rows(
            [draft],
            [source],
            concept_ids,
            minimum_span_codepoints=policy["evidence_span_contract"][
                "minimum_codepoints_per_positive_label"
            ],
            minimum_confidence_ppm=policy["confidence_contract"][
                "minimum_confidence_ppm"
            ],
        )
    except AuthoredReviewCompileError as error:
        raise AuthoredModelReviewError(str(error)) from error
    return draft


def _tensor_versions(model: Any) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, value._version)
        for name, value in sorted(model.state_dict(keep_vars=True).items())
    )


def _rng_state(torch: Any) -> tuple[bytes, bytes]:
    return (
        bytes(torch.random.get_rng_state().tolist()),
        bytes(torch.cuda.get_rng_state(0).tolist()),
    )


def _load_model(
    model_root: Path,
    *,
    manifest: Path,
    restoration_receipt: Path,
    reviewer: str,
) -> tuple[Any, Any, dict[str, Any]]:
    if reviewer not in REVIEWERS:
        raise AuthoredModelReviewError("reviewer identity differs")
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise AuthoredModelReviewError("model-review runtime is unavailable") from error
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise AuthoredModelReviewError("exactly one H100 BF16 GPU is required")
    try:
        snapshot = validate_external_snapshot(
            model_root,
            manifest_path=manifest,
            receipt_path=restoration_receipt,
            spec=REVIEWERS[reviewer],
        )
    except ExternalSnapshotError as error:
        raise AuthoredModelReviewError("reviewer snapshot differs") from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        output_loading_info=True,
    )
    if not isinstance(loading, dict):
        raise AuthoredModelReviewError("reviewer model load differs")
    normalized_loading = {}
    for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs"):
        values = loading.get(key)
        if not isinstance(values, (list, tuple, set)) or any(
            not isinstance(value, str) for value in values
        ):
            raise AuthoredModelReviewError("reviewer model load differs")
        normalized_loading[key] = sorted(values)
    if (
        normalized_loading["missing_keys"]
        or normalized_loading["mismatched_keys"]
        or normalized_loading["error_msgs"]
        or (reviewer == "smollm3_3b" and normalized_loading["unexpected_keys"])
        or (
            reviewer == "qwen35_9b"
            and any(
                not (name.startswith("model.visual.") or name.startswith("mtp."))
                for name in normalized_loading["unexpected_keys"]
            )
        )
    ):
        raise AuthoredModelReviewError("reviewer model load differs")
    context = getattr(model.config, "max_position_embeddings", None)
    if (
        isinstance(context, bool)
        or not isinstance(context, int)
        or context < MAX_INPUT_TOKENS + MAX_NEW_TOKENS
        or tokenizer.eos_token_id is None
    ):
        raise AuthoredModelReviewError("reviewer context or tokenizer differs")
    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    if not parameters or any(
        tensor.device.type != "cuda" or tensor.device.index not in {0, None}
        for _, tensor in [*parameters, *buffers]
    ):
        raise AuthoredModelReviewError("reviewer model residency differs")
    source = Path(inspect.getsourcefile(type(model)) or "")
    return (
        model,
        tokenizer,
        {
            "reviewer": reviewer,
            "snapshot": snapshot,
            "model_class": type(model).__name__,
            "parameter_count": sum(value.numel() for _, value in parameters),
            "maximum_position_embeddings": context,
            "tokenizer_length": len(tokenizer),
            "unexpected_weight_count": len(normalized_loading["unexpected_keys"]),
            "unexpected_weights_sha256": canonical_sha256(
                normalized_loading["unexpected_keys"]
            ),
            "transformers_version": transformers.__version__,
            "model_source_sha256": sha256_file(source),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
    )


def _messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object grounded only in verbatim chapter "
                "text. End immediately after its closing brace. Never add analysis, "
                "notes, markdown, or commentary."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _input_ids(encoded: Any, torch: Any) -> Any:
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
    elif (
        isinstance(encoded, Mapping)
        and set(encoded) in ({"input_ids"}, {"input_ids", "attention_mask"})
        and isinstance(encoded["input_ids"], torch.Tensor)
    ):
        input_ids = encoded["input_ids"]
        if "attention_mask" in encoded and (
            not isinstance(encoded["attention_mask"], torch.Tensor)
            or encoded["attention_mask"].shape != input_ids.shape
            or not torch.equal(encoded["attention_mask"], torch.ones_like(input_ids))
        ):
            raise AuthoredModelReviewError("review prompt tokenization differs")
    else:
        raise AuthoredModelReviewError("review prompt tokenization differs")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] <= 0:
        raise AuthoredModelReviewError("review prompt tokenization differs")
    return input_ids


def _generate(model: Any, tokenizer: Any, prompt: str) -> tuple[str, int, int]:
    import torch

    encoded = tokenizer.apply_chat_template(
        _messages(prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    encoded = _input_ids(encoded, torch)
    input_tokens = int(encoded.shape[1])
    if input_tokens <= 0 or input_tokens > MAX_INPUT_TOKENS:
        raise AuthoredModelReviewError("review prompt exceeds frozen context budget")
    encoded = encoded.to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=encoded,
            attention_mask=torch.ones_like(encoded),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, input_tokens:]
    return (
        tokenizer.decode(continuation, skip_special_tokens=True),
        input_tokens,
        int(continuation.numel()),
    )


def _raw_path(root: Path, index: int) -> Path:
    return root / "raw" / f"{index:03d}.json"


def _failure_path(root: Path, index: int) -> Path:
    return root / "raw" / f"{index:03d}.failure.json"


def _validate_attempts(
    attempts: Any,
    *,
    source: dict[str, Any],
    concept_ids: set[str],
    concept_prompt: str,
    policy: dict[str, Any],
    require_accepted_final: bool,
) -> dict[str, Any] | None:
    if (
        not isinstance(attempts, list)
        or not 1 <= len(attempts) <= MAX_ATTEMPTS
        or (not require_accepted_final and len(attempts) != MAX_ATTEMPTS)
    ):
        raise AuthoredModelReviewError("resumed raw attempts differ")
    repair = None
    accepted_draft = None
    for offset, attempt in enumerate(attempts, 1):
        if (
            not isinstance(attempt, dict)
            or set(attempt) != _ATTEMPT_KEYS
            or attempt["attempt"] != offset
            or not isinstance(attempt["response"], str)
            or attempt["response_sha256"] != _sha256(attempt["response"].encode())
            or isinstance(attempt["input_tokens"], bool)
            or not isinstance(attempt["input_tokens"], int)
            or not 0 < attempt["input_tokens"] <= MAX_INPUT_TOKENS
            or isinstance(attempt["output_tokens"], bool)
            or not isinstance(attempt["output_tokens"], int)
            or not 0 < attempt["output_tokens"] <= MAX_NEW_TOKENS
            or not isinstance(attempt["accepted"], bool)
        ):
            raise AuthoredModelReviewError("resumed raw attempt differs")
        prompt = _prompt(source, concept_prompt, repair)
        if attempt["prompt_sha256"] != _sha256(prompt.encode()):
            raise AuthoredModelReviewError("resumed raw prompt differs")
        try:
            candidate = _draft_attempt(
                source,
                attempt["response"],
                attempt["output_tokens"],
                concept_ids,
                policy,
            )
            error = None
        except AuthoredModelReviewError as caught:
            candidate = None
            error = str(caught)
        if attempt["accepted"]:
            if (
                not require_accepted_final
                or error is not None
                or attempt["rejection"] is not None
                or offset != len(attempts)
            ):
                raise AuthoredModelReviewError("resumed accepted attempt differs")
            accepted_draft = candidate
        else:
            if (
                error is None
                or attempt["rejection"] != error
                or (require_accepted_final and offset == len(attempts))
            ):
                raise AuthoredModelReviewError("resumed rejected attempt differs")
            repair = error
    if require_accepted_final and accepted_draft is None:
        raise AuthoredModelReviewError("resumed accepted attempt differs")
    if not require_accepted_final and accepted_draft is not None:
        raise AuthoredModelReviewError("resumed failure attempt differs")
    return accepted_draft


def _draft_attempt(
    source: dict[str, Any],
    response: str,
    output_tokens: int,
    concept_ids: set[str],
    policy: dict[str, Any],
) -> dict[str, Any]:
    try:
        return _draft_from_response(source, response, concept_ids, policy)
    except AuthoredModelReviewError as error:
        if output_tokens == MAX_NEW_TOKENS:
            raise AuthoredModelReviewError(
                "model response exhausted the frozen output budget; return a much "
                "smaller exact JSON object with only direct concepts"
            ) from error
        raise


def _failure_payload(
    *,
    source: dict[str, Any],
    index: int,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": FAILURE_SCHEMA,
        "status": "exhausted_frozen_attempts",
        "index": index,
        "review_identity_sha256": source["review_identity_sha256"],
        "source_text_sha256": source["text_sha256"],
        "attempts": attempts,
        "human_review_completed": False,
        "audit_qualified": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _validate_failure(
    failure: Any,
    *,
    source: dict[str, Any],
    index: int,
    concept_ids: set[str],
    concept_prompt: str,
    policy: dict[str, Any],
) -> None:
    if (
        not isinstance(failure, dict)
        or set(failure) != _FAILURE_KEYS
        or failure["schema"] != FAILURE_SCHEMA
        or failure["status"] != "exhausted_frozen_attempts"
        or failure["index"] != index
        or failure["review_identity_sha256"] != source["review_identity_sha256"]
        or failure["source_text_sha256"] != source["text_sha256"]
        or failure["human_review_completed"] is not False
        or failure["audit_qualified"] is not False
        or failure["training_authorized"] is not False
        or failure["four_b_training_authorized"] is not False
        or failure["receipt_sha256"]
        != canonical_sha256(
            {key: value for key, value in failure.items() if key != "receipt_sha256"}
        )
    ):
        raise AuthoredModelReviewError("preserved review failure differs")
    _validate_attempts(
        failure["attempts"],
        source=source,
        concept_ids=concept_ids,
        concept_prompt=concept_prompt,
        policy=policy,
        require_accepted_final=False,
    )


def _validate_raw(
    raw: Any,
    *,
    source: dict[str, Any],
    index: int,
    concept_ids: set[str],
    concept_prompt: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(raw, dict)
        or set(raw) != _RAW_KEYS
        or raw["schema"] != RAW_SCHEMA
        or raw["index"] != index
        or raw["review_identity_sha256"] != source["review_identity_sha256"]
        or raw["source_text_sha256"] != source["text_sha256"]
        or not isinstance(raw["draft"], dict)
    ):
        raise AuthoredModelReviewError("resumed raw review differs")
    accepted_draft = _validate_attempts(
        raw["attempts"],
        source=source,
        concept_ids=concept_ids,
        concept_prompt=concept_prompt,
        policy=policy,
        require_accepted_final=True,
    )
    if accepted_draft is None or raw["draft"] != accepted_draft:
        raise AuthoredModelReviewError("resumed raw draft differs")
    return accepted_draft


def _identity_payload(
    *,
    reviewer: str,
    runtime: dict[str, Any],
    inputs: _Inputs,
    source_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "sai-authored-curriculum-model-reviewer-identity-v1",
        "status": "candidate_model_reviewer",
        "reviewer": reviewer,
        "runtime": runtime,
        "blind_review_packet_sha256": _sha256(inputs.packet_encoded),
        "blind_review_packet_receipt_sha256": _sha256(inputs.packet_receipt_encoded),
        "concept_list_sha256": _sha256(inputs.concept_encoded),
        "annotation_policy_sha256": _sha256(inputs.policy_encoded),
        "runner_source_sha256": source_sha256,
        "generation": {
            "maximum_input_tokens": MAX_INPUT_TOKENS,
            "maximum_new_tokens": MAX_NEW_TOKENS,
            "maximum_attempts": MAX_ATTEMPTS,
            "do_sample": False,
            "num_beams": 1,
            "thinking_enabled": False,
        },
        "hidden_review_key_accessed": False,
        "human_review_completed": False,
        "audit_qualified": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _result_payload(
    *,
    identity: dict[str, Any],
    drafts: list[dict[str, Any]],
    raw_hashes: list[dict[str, str]],
    draft_encoded: bytes,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_candidate_model_review",
        "reviewer_identity_sha256": identity["receipt_sha256"],
        "rows": len(drafts),
        "raw_rows_sha256": canonical_sha256(raw_hashes),
        "draft_sha256": _sha256(draft_encoded),
        "model_state_unchanged": True,
        "rng_state_unchanged": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "hidden_review_key_accessed": False,
        "human_review_completed": False,
        "audit_qualified": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
        "limitations": [
            "model_labels_are_candidate_triage_evidence_only",
            "model_agreement_does_not_replace_independent_human_review",
            "no_data_admission_training_or_architecture_promotion_is_authorized",
        ],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_result(
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
    output_root: Path,
) -> dict[str, Any]:
    """Replay the completed candidate review without loading or executing a model."""

    if reviewer not in REVIEWERS:
        raise AuthoredModelReviewError("reviewer identity differs")
    inputs = _blind_inputs(
        review_packet=review_packet,
        review_packet_receipt=review_packet_receipt,
        expected_review_packet_sha256=expected_review_packet_sha256,
        expected_review_packet_receipt_sha256=expected_review_packet_receipt_sha256,
        concept_list=concept_list,
        annotation_policy=annotation_policy,
    )
    try:
        snapshot = validate_external_snapshot(
            model_root,
            manifest_path=manifest,
            receipt_path=restoration_receipt,
            spec=REVIEWERS[reviewer],
        )
    except ExternalSnapshotError as error:
        raise AuthoredModelReviewError("reviewer snapshot differs") from error
    output_root = output_root.resolve()
    raw_root = output_root / "raw"
    expected_names = {"reviewer-identity.json", "draft.jsonl", "receipt.json", "raw"}
    if (
        not output_root.is_dir()
        or output_root.is_symlink()
        or stat.S_IMODE(output_root.stat().st_mode) & 0o222
        or {path.name for path in output_root.iterdir()} != expected_names
        or not raw_root.is_dir()
        or raw_root.is_symlink()
        or stat.S_IMODE(raw_root.stat().st_mode) & 0o222
    ):
        raise AuthoredModelReviewError("completed review tree differs")
    identity, identity_encoded = _json(
        output_root / "reviewer-identity.json", "reviewer identity", 1 << 20
    )
    draft_encoded = _read_regular_bytes(
        output_root / "draft.jsonl", maximum_bytes=1 << 30
    )
    receipt, receipt_encoded = _json(
        output_root / "receipt.json", "model review receipt", 1 << 20
    )
    if (
        not isinstance(identity, dict)
        or set(identity) != _IDENTITY_KEYS
        or not isinstance(identity.get("runtime"), dict)
        or set(identity["runtime"]) != _RUNTIME_KEYS
        or identity["schema"] != "sai-authored-curriculum-model-reviewer-identity-v1"
        or identity["status"] != "candidate_model_reviewer"
        or identity["reviewer"] != reviewer
        or identity["runtime"]["reviewer"] != reviewer
        or identity["runtime"].get("snapshot") != snapshot
        or not isinstance(identity["runtime"]["gpu_name"], str)
        or "H100" not in identity["runtime"]["gpu_name"]
        or identity["blind_review_packet_sha256"] != _sha256(inputs.packet_encoded)
        or identity["blind_review_packet_receipt_sha256"]
        != _sha256(inputs.packet_receipt_encoded)
        or identity["concept_list_sha256"] != _sha256(inputs.concept_encoded)
        or identity["annotation_policy_sha256"] != _sha256(inputs.policy_encoded)
        or identity["runner_source_sha256"] != sha256_file(Path(__file__))
        or identity["generation"]
        != {
            "maximum_input_tokens": MAX_INPUT_TOKENS,
            "maximum_new_tokens": MAX_NEW_TOKENS,
            "maximum_attempts": MAX_ATTEMPTS,
            "do_sample": False,
            "num_beams": 1,
            "thinking_enabled": False,
        }
        or any(
            identity[key] is not False
            for key in (
                "hidden_review_key_accessed",
                "human_review_completed",
                "audit_qualified",
                "training_authorized",
                "four_b_training_authorized",
            )
        )
        or identity["receipt_sha256"]
        != canonical_sha256(
            {key: value for key, value in identity.items() if key != "receipt_sha256"}
        )
        or identity_encoded
        != json.dumps(identity, sort_keys=True, indent=2).encode() + b"\n"
    ):
        raise AuthoredModelReviewError("reviewer identity replay differs")
    raw_paths = sorted(raw_root.glob("*.json"))
    if len(raw_paths) != len(inputs.packet) or {
        path.name for path in raw_root.iterdir()
    } != {f"{index:03d}.json" for index in range(len(inputs.packet))}:
        raise AuthoredModelReviewError("completed raw population differs")
    for path in [
        output_root / "reviewer-identity.json",
        output_root / "draft.jsonl",
        output_root / "receipt.json",
        *raw_paths,
    ]:
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise AuthoredModelReviewError("completed review member is writable")
    concept_ids = {item["concept_id"] for item in inputs.concept_payload["concepts"]}
    concept_prompt = _concept_prompt(inputs.concept_payload)
    drafts = []
    raw_hashes = []
    for index, (path, source) in enumerate(zip(raw_paths, inputs.packet, strict=True)):
        raw, raw_encoded = _json(path, "completed raw review", 8 << 20)
        drafts.append(
            _validate_raw(
                raw,
                source=source,
                index=index,
                concept_ids=concept_ids,
                concept_prompt=concept_prompt,
                policy=inputs.policy,
            )
        )
        raw_hashes.append({"path": path.name, "sha256": _sha256(raw_encoded)})
    expected_draft = b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
        for row in drafts
    )
    expected_receipt = _result_payload(
        identity=identity,
        drafts=drafts,
        raw_hashes=raw_hashes,
        draft_encoded=expected_draft,
    )
    if (
        draft_encoded != expected_draft
        or receipt != expected_receipt
        or receipt_encoded
        != json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n"
    ):
        raise AuthoredModelReviewError("completed review replay differs")
    return receipt


def run(
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
    output_root: Path,
) -> dict[str, Any]:
    """Generate or resume one complete candidate model-review population."""

    inputs = _blind_inputs(
        review_packet=review_packet,
        review_packet_receipt=review_packet_receipt,
        expected_review_packet_sha256=expected_review_packet_sha256,
        expected_review_packet_receipt_sha256=expected_review_packet_receipt_sha256,
        concept_list=concept_list,
        annotation_policy=annotation_policy,
    )
    output_root = output_root.resolve()
    final_draft = output_root / "draft.jsonl"
    final_receipt = output_root / "receipt.json"
    if final_draft.exists() or final_receipt.exists():
        if not (final_draft.exists() and final_receipt.exists()):
            raise AuthoredModelReviewError("review final output is incomplete")
        return validate_result(
            reviewer=reviewer,
            model_root=model_root,
            manifest=manifest,
            restoration_receipt=restoration_receipt,
            review_packet=review_packet,
            review_packet_receipt=review_packet_receipt,
            expected_review_packet_sha256=expected_review_packet_sha256,
            expected_review_packet_receipt_sha256=(
                expected_review_packet_receipt_sha256
            ),
            concept_list=concept_list,
            annotation_policy=annotation_policy,
            output_root=output_root,
        )
    model, tokenizer, runtime = _load_model(
        model_root,
        manifest=manifest,
        restoration_receipt=restoration_receipt,
        reviewer=reviewer,
    )
    source_sha256 = sha256_file(Path(__file__))
    identity = _identity_payload(
        reviewer=reviewer,
        runtime=runtime,
        inputs=inputs,
        source_sha256=source_sha256,
    )
    identity_encoded = json.dumps(identity, sort_keys=True, indent=2).encode() + b"\n"
    identity_path = output_root / "reviewer-identity.json"
    raw_root = output_root / "raw"
    if not output_root.exists():
        output_root.mkdir(mode=0o700)
        raw_root.mkdir(mode=0o700)
        _write_create_only(identity_path, identity_encoded)
    elif (
        not output_root.is_dir()
        or output_root.is_symlink()
        or not raw_root.is_dir()
        or raw_root.is_symlink()
        or _read_regular_bytes(identity_path, maximum_bytes=1 << 20) != identity_encoded
    ):
        raise AuthoredModelReviewError("review output identity differs")
    concept_ids = {item["concept_id"] for item in inputs.concept_payload["concepts"]}
    concept_prompt = _concept_prompt(inputs.concept_payload)
    versions = _tensor_versions(model)
    import torch

    rng = _rng_state(torch)
    model.eval()
    drafts = []
    raw_hashes = []
    for index, source in enumerate(inputs.packet):
        path = _raw_path(output_root, index)
        failure_path = _failure_path(output_root, index)
        if failure_path.exists():
            failure, _ = _json(failure_path, "preserved review failure", 8 << 20)
            _validate_failure(
                failure,
                source=source,
                index=index,
                concept_ids=concept_ids,
                concept_prompt=concept_prompt,
                policy=inputs.policy,
            )
            raise AuthoredModelReviewError(
                f"review row {index} has a preserved exhausted-attempt failure"
            )
        if path.exists():
            raw, raw_encoded = _json(path, "resumed raw review", 8 << 20)
            draft = _validate_raw(
                raw,
                source=source,
                index=index,
                concept_ids=concept_ids,
                concept_prompt=concept_prompt,
                policy=inputs.policy,
            )
        else:
            attempts = []
            repair = None
            draft = None
            for attempt in range(MAX_ATTEMPTS):
                prompt = _prompt(source, concept_prompt, repair)
                response, input_tokens, output_tokens = _generate(
                    model, tokenizer, prompt
                )
                record = {
                    "attempt": attempt + 1,
                    "prompt_sha256": _sha256(prompt.encode()),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "response": response,
                    "response_sha256": _sha256(response.encode()),
                }
                try:
                    draft = _draft_attempt(
                        source,
                        response,
                        output_tokens,
                        concept_ids,
                        inputs.policy,
                    )
                    record["accepted"] = True
                    record["rejection"] = None
                    attempts.append(record)
                    break
                except AuthoredModelReviewError as error:
                    record["accepted"] = False
                    record["rejection"] = str(error)
                    attempts.append(record)
                    repair = str(error)
            if draft is None:
                failure = _failure_payload(
                    source=source,
                    index=index,
                    attempts=attempts,
                )
                failure_encoded = (
                    json.dumps(failure, sort_keys=True, indent=2).encode() + b"\n"
                )
                _write_create_only(failure_path, failure_encoded)
                os.chmod(failure_path, 0o400)
                raise AuthoredModelReviewError(
                    f"review row {index} exhausted frozen attempts; "
                    f"evidence preserved at {failure_path}"
                )
            raw = {
                "schema": RAW_SCHEMA,
                "index": index,
                "review_identity_sha256": source["review_identity_sha256"],
                "source_text_sha256": source["text_sha256"],
                "attempts": attempts,
                "draft": draft,
            }
            raw_encoded = json.dumps(raw, sort_keys=True, indent=2).encode() + b"\n"
            _write_create_only(path, raw_encoded)
        drafts.append(draft)
        raw_hashes.append({"path": path.name, "sha256": _sha256(raw_encoded)})
    if _tensor_versions(model) != versions or _rng_state(torch) != rng:
        raise AuthoredModelReviewError("model review mutated model or RNG state")
    draft_encoded = b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
        for row in drafts
    )
    receipt = _result_payload(
        identity=identity,
        drafts=drafts,
        raw_hashes=raw_hashes,
        draft_encoded=draft_encoded,
    )
    receipt_encoded = json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n"
    created = []
    try:
        _write_create_only(final_draft, draft_encoded)
        created.append(final_draft)
        _write_create_only(final_receipt, receipt_encoded)
        created.append(final_receipt)
        os.chmod(raw_root, 0o500)
        os.chmod(output_root, 0o500)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return receipt


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
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-review-packet-sha256", required=True)
    parser.add_argument("--expected-review-packet-receipt-sha256", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    function = run if command == "generate" else validate_result
    payload = function(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["rows"],
                "receipt_sha256": payload["receipt_sha256"],
                "audit_qualified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

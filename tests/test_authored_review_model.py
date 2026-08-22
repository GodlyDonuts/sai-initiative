from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sai.data.authored_review_model as model_review
from sai.data.authored_review_model import (
    AuthoredModelReviewError,
    _blind_inputs,
    _draft_from_response,
    _failure_payload,
    _identity_payload,
    _input_ids,
    _prompt,
    _response_object,
    _result_payload,
    _validate_failure,
    _validate_raw,
    validate_result,
)

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts" / "authored-curriculum-sources-r1"
PACKET = ARTIFACT / "authored-curriculum-blind-review.jsonl"
PACKET_RECEIPT = ARTIFACT / "authored-curriculum-review-receipt.json"
CONCEPTS = ROOT / "docs" / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json"
POLICY = ROOT / "docs" / "SAI_SEMANTIC_ANNOTATION_POLICY.json"
JOB = ROOT / "jobs" / "sai-authored-model-review-single-h100.sbatch"


def _inputs():
    return _blind_inputs(
        review_packet=PACKET,
        review_packet_receipt=PACKET_RECEIPT,
        expected_review_packet_sha256=hashlib.sha256(PACKET.read_bytes()).hexdigest(),
        expected_review_packet_receipt_sha256=hashlib.sha256(
            PACKET_RECEIPT.read_bytes()
        ).hexdigest(),
        concept_list=CONCEPTS,
        annotation_policy=POLICY,
    )


def test_blind_inputs_never_require_hidden_key() -> None:
    inputs = _inputs()
    assert len(inputs.packet) == 127
    assert all("source_path" not in row for row in inputs.packet)
    assert all("candidate_stage" not in row for row in inputs.packet)
    assert all("required_prior_concepts" not in row for row in inputs.packet)


def test_prompt_contains_blind_text_and_no_hidden_metadata() -> None:
    inputs = _inputs()
    prompt = _prompt(inputs.packet[0], "- code.literal: Literal values", None)
    assert inputs.packet[0]["text"] in prompt
    assert "source_path" not in prompt
    assert "candidate_stage" not in prompt
    assert "required_prior_concepts" not in prompt


def test_response_compiles_exact_quote() -> None:
    inputs = _inputs()
    source = inputs.packet[0]
    quote = next(
        line
        for line in source["text"].splitlines()
        if len(line) >= 24 and source["text"].count(line) == 1
    )
    response = json.dumps(
        {
            "instructional_quality_ppm": 900_000,
            "assumed_prior_concepts": [],
            "taught_concepts": [
                {
                    "concept_id": "code.literal",
                    "confidence_ppm": 900_000,
                    "evidence_quotes": [quote],
                }
            ],
            "defects": [],
            "admission_recommendation": "admit",
        }
    )
    concepts = {item["concept_id"] for item in inputs.concept_payload["concepts"]}
    draft = _draft_from_response(source, response, concepts, inputs.policy)
    assert draft["taught_concepts"][0]["evidence_quotes"] == [quote]


def test_input_ids_accepts_exact_tensor_or_input_only_mapping() -> None:
    import torch

    tensor = torch.tensor([[1, 2, 3]])
    assert _input_ids(tensor, torch) is tensor
    assert _input_ids({"input_ids": tensor}, torch) is tensor
    assert (
        _input_ids(
            {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}, torch
        )
        is tensor
    )
    with pytest.raises(AuthoredModelReviewError, match="tokenization differs"):
        _input_ids(
            {"input_ids": tensor, "attention_mask": torch.tensor([[1, 1, 0]])},
            torch,
        )
    with pytest.raises(AuthoredModelReviewError, match="tokenization differs"):
        _input_ids(
            {"input_ids": tensor, "attention_mask": tensor, "other": tensor}, torch
        )
    with pytest.raises(AuthoredModelReviewError, match="tokenization differs"):
        _input_ids(torch.tensor([1, 2, 3]), torch)


def test_response_rejects_commentary_and_missing_quote() -> None:
    with pytest.raises(AuthoredModelReviewError, match="JSON"):
        _response_object('Commentary {"instructional_quality_ppm": 1}')
    inputs = _inputs()
    source = inputs.packet[0]
    concepts = {item["concept_id"] for item in inputs.concept_payload["concepts"]}
    response = json.dumps(
        {
            "instructional_quality_ppm": 900_000,
            "assumed_prior_concepts": [],
            "taught_concepts": [
                {
                    "concept_id": "code.literal",
                    "confidence_ppm": 900_000,
                    "evidence_quotes": ["this quote does not exist anywhere"],
                }
            ],
            "defects": [],
            "admission_recommendation": "admit",
        }
    )
    with pytest.raises(AuthoredModelReviewError, match="missing or ambiguous"):
        _draft_from_response(source, response, concepts, inputs.policy)


def test_blind_inputs_reject_packet_hash_mismatch() -> None:
    with pytest.raises(AuthoredModelReviewError, match="receipt differs"):
        _blind_inputs(
            review_packet=PACKET,
            review_packet_receipt=PACKET_RECEIPT,
            expected_review_packet_sha256="0" * 64,
            expected_review_packet_receipt_sha256=hashlib.sha256(
                PACKET_RECEIPT.read_bytes()
            ).hexdigest(),
            concept_list=CONCEPTS,
            annotation_policy=POLICY,
        )


def test_raw_resume_replays_repair_chain_and_rejects_tamper() -> None:
    inputs = _inputs()
    source = inputs.packet[0]
    concept_ids = {item["concept_id"] for item in inputs.concept_payload["concepts"]}
    concept_prompt = "- code.literal: Literal values"
    quote = next(
        line
        for line in source["text"].splitlines()
        if len(line) >= 24 and source["text"].count(line) == 1
    )
    bad_response = "not JSON"
    rejection = "model response is not exact JSON"
    repaired = json.dumps(
        {
            "instructional_quality_ppm": 900_000,
            "assumed_prior_concepts": [],
            "taught_concepts": [
                {
                    "concept_id": "code.literal",
                    "confidence_ppm": 900_000,
                    "evidence_quotes": [quote],
                }
            ],
            "defects": [],
            "admission_recommendation": "admit",
        }
    )
    draft = _draft_from_response(source, repaired, concept_ids, inputs.policy)
    first_prompt = _prompt(source, concept_prompt, None)
    second_prompt = _prompt(source, concept_prompt, rejection)
    raw = {
        "schema": "sai-authored-curriculum-model-review-raw-row-v1",
        "index": 0,
        "review_identity_sha256": source["review_identity_sha256"],
        "source_text_sha256": source["text_sha256"],
        "attempts": [
            {
                "attempt": 1,
                "prompt_sha256": hashlib.sha256(first_prompt.encode()).hexdigest(),
                "input_tokens": 100,
                "output_tokens": 10,
                "response": bad_response,
                "response_sha256": hashlib.sha256(bad_response.encode()).hexdigest(),
                "accepted": False,
                "rejection": rejection,
            },
            {
                "attempt": 2,
                "prompt_sha256": hashlib.sha256(second_prompt.encode()).hexdigest(),
                "input_tokens": 110,
                "output_tokens": 20,
                "response": repaired,
                "response_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
                "accepted": True,
                "rejection": None,
            },
        ],
        "draft": draft,
    }
    assert (
        _validate_raw(
            raw,
            source=source,
            index=0,
            concept_ids=concept_ids,
            concept_prompt=concept_prompt,
            policy=inputs.policy,
        )
        == draft
    )
    raw["attempts"][0]["rejection"] = "different rejection"
    with pytest.raises(AuthoredModelReviewError, match="rejected attempt differs"):
        _validate_raw(
            raw,
            source=source,
            index=0,
            concept_ids=concept_ids,
            concept_prompt=concept_prompt,
            policy=inputs.policy,
        )


def test_exhausted_failure_preserves_and_replays_every_rejected_attempt() -> None:
    inputs = _inputs()
    source = inputs.packet[0]
    concept_ids = {item["concept_id"] for item in inputs.concept_payload["concepts"]}
    concept_prompt = model_review._concept_prompt(inputs.concept_payload)
    attempts = []
    repair = None
    for attempt in range(model_review.MAX_ATTEMPTS):
        prompt = _prompt(source, concept_prompt, repair)
        response = "not JSON"
        repair = "model response is not exact JSON"
        attempts.append(
            {
                "attempt": attempt + 1,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_tokens": 100,
                "output_tokens": 2,
                "response": response,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "accepted": False,
                "rejection": repair,
            }
        )
    failure = _failure_payload(source=source, index=0, attempts=attempts)
    _validate_failure(
        failure,
        source=source,
        index=0,
        concept_ids=concept_ids,
        concept_prompt=concept_prompt,
        policy=inputs.policy,
    )
    failure["attempts"][0]["response"] = "mutated"
    with pytest.raises(AuthoredModelReviewError, match="failure differs"):
        _validate_failure(
            failure,
            source=source,
            index=0,
            concept_ids=concept_ids,
            concept_prompt=concept_prompt,
            policy=inputs.policy,
        )
    failure["receipt_sha256"] = model_review.canonical_sha256(
        {key: value for key, value in failure.items() if key != "receipt_sha256"}
    )
    with pytest.raises(AuthoredModelReviewError, match="attempt differs"):
        _validate_failure(
            failure,
            source=source,
            index=0,
            concept_ids=concept_ids,
            concept_prompt=concept_prompt,
            policy=inputs.policy,
        )


def test_completed_candidate_review_replays_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs()
    snapshot = {
        "repository": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "tree_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "receipt_file_sha256": "3" * 64,
        "file_count": 1,
        "total_bytes": 1,
        "files": [{"path": "config.json", "bytes": 1, "sha256": "4" * 64}],
    }
    monkeypatch.setattr(
        model_review,
        "validate_external_snapshot",
        lambda *args, **kwargs: snapshot,
    )
    runtime = {
        "reviewer": "qwen35_9b",
        "snapshot": snapshot,
        "model_class": "FakeReviewer",
        "parameter_count": 1,
        "maximum_position_embeddings": 65_536,
        "tokenizer_length": 100,
        "unexpected_weight_count": 0,
        "unexpected_weights_sha256": "6" * 64,
        "transformers_version": "test",
        "model_source_sha256": "5" * 64,
        "python": "test",
        "torch": "test",
        "cuda": "test",
        "gpu_name": "H100 test",
        "gpu_capability": [9, 0],
    }
    identity = _identity_payload(
        reviewer="qwen35_9b",
        runtime=runtime,
        inputs=inputs,
        source_sha256=hashlib.sha256(
            Path(model_review.__file__).read_bytes()
        ).hexdigest(),
    )
    root = tmp_path / "result"
    raw_root = root / "raw"
    raw_root.mkdir(parents=True)
    concept_ids = {item["concept_id"] for item in inputs.concept_payload["concepts"]}
    concept_prompt = model_review._concept_prompt(inputs.concept_payload)
    drafts = []
    raw_hashes = []
    for index, source in enumerate(inputs.packet):
        quote = next(
            source["text"][start : start + 16]
            for start in range(len(source["text"]) - 15)
            if source["text"].count(source["text"][start : start + 16]) == 1
        )
        response = json.dumps(
            {
                "instructional_quality_ppm": 900_000,
                "assumed_prior_concepts": [],
                "taught_concepts": [
                    {
                        "concept_id": "code.literal",
                        "confidence_ppm": 900_000,
                        "evidence_quotes": [quote],
                    }
                ],
                "defects": [],
                "admission_recommendation": "admit",
            }
        )
        draft = _draft_from_response(source, response, concept_ids, inputs.policy)
        prompt = _prompt(source, concept_prompt, None)
        raw = {
            "schema": "sai-authored-curriculum-model-review-raw-row-v1",
            "index": index,
            "review_identity_sha256": source["review_identity_sha256"],
            "source_text_sha256": source["text_sha256"],
            "attempts": [
                {
                    "attempt": 1,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "accepted": True,
                    "rejection": None,
                }
            ],
            "draft": draft,
        }
        encoded = json.dumps(raw, sort_keys=True, indent=2).encode() + b"\n"
        path = raw_root / f"{index:03d}.json"
        path.write_bytes(encoded)
        path.chmod(0o444)
        drafts.append(draft)
        raw_hashes.append(
            {"path": path.name, "sha256": hashlib.sha256(encoded).hexdigest()}
        )
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
    artifacts = {
        "reviewer-identity.json": json.dumps(
            identity, sort_keys=True, indent=2
        ).encode()
        + b"\n",
        "draft.jsonl": draft_encoded,
        "receipt.json": json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n",
    }
    for name, encoded in artifacts.items():
        path = root / name
        path.write_bytes(encoded)
        path.chmod(0o444)
    raw_root.chmod(0o500)
    root.chmod(0o500)
    assert (
        validate_result(
            reviewer="qwen35_9b",
            model_root=tmp_path / "model",
            manifest=tmp_path / "manifest.json",
            restoration_receipt=tmp_path / "restoration.json",
            review_packet=PACKET,
            review_packet_receipt=PACKET_RECEIPT,
            expected_review_packet_sha256=hashlib.sha256(
                PACKET.read_bytes()
            ).hexdigest(),
            expected_review_packet_receipt_sha256=hashlib.sha256(
                PACKET_RECEIPT.read_bytes()
            ).hexdigest(),
            concept_list=CONCEPTS,
            annotation_policy=POLICY,
            output_root=root,
        )
        == receipt
    )


def test_model_review_job_is_one_gpu_blind_and_nontraining() -> None:
    script = JOB.read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in script
    assert "#SBATCH --no-requeue" in script
    assert "generate" in script
    assert '"$REVIEWER" == qwen35_9b' in script
    assert '"$REVIEWER" == smollm3_3b' in script
    assert "REVIEW_KEY" not in script
    assert "candidate-stage" not in script
    assert "optimizer" not in script.lower()
    assert "torchrun" not in script
    assert 'export PYTHONPATH="$SAI_ROOT/src"' in script
    assert "export GIT_OPTIONAL_LOCKS=0" in script
    assert 'status --short)"' in script

"""Bind every admitted Sai training component into one signed release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_component_admission import SCHEMA as BRIDGE_ADMISSION_SCHEMA
from sai.data.bridge_component_hf_publish import SCHEMA as BRIDGE_PUBLICATION_SCHEMA
from sai.data.common_pile_stack_edu_practical_admission import (
    SCHEMA as CODE_ADMISSION_SCHEMA,
)
from sai.data.common_pile_stack_edu_practical_hf_publish import (
    METADATA_SCHEMA as CODE_PUBLICATION_SCHEMA,
)
from sai.data.practical_corpus_audit import SCHEMA as FOUNDATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-final-training-release-v1"


class FinalTrainingReleaseError(RuntimeError):
    """A required component, lineage edge, or split invariant differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise FinalTrainingReleaseError("signed release input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalTrainingReleaseError("signed release input differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise FinalTrainingReleaseError("signed release input differs")
    return payload


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FinalTrainingReleaseError(f"{label} differs")
    return value


def build_release(
    foundation_path: Path,
    code_admission_path: Path,
    code_publication_path: Path,
    bridge_admission_path: Path,
    bridge_publication_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Fail closed unless the exact three-component training stream is complete."""

    if output.exists() or output.is_symlink():
        raise FinalTrainingReleaseError("release output exists")
    foundation = _load_signed(foundation_path, FOUNDATION_SCHEMA)
    code = _load_signed(code_admission_path, CODE_ADMISSION_SCHEMA)
    code_publication = _load_signed(code_publication_path, CODE_PUBLICATION_SCHEMA)
    bridge = _load_signed(bridge_admission_path, BRIDGE_ADMISSION_SCHEMA)
    bridge_publication = _load_signed(
        bridge_publication_path, BRIDGE_PUBLICATION_SCHEMA
    )

    foundation_bytes = _positive(
        foundation.get("totals", {}).get("text_utf8_bytes"), "foundation bytes"
    )
    foundation_rows = _positive(
        foundation.get("totals", {}).get("rows"), "foundation rows"
    )
    code_bytes = _positive(
        code.get("counts", {}).get("admitted_text_utf8_bytes"), "code bytes"
    )
    code_rows = _positive(code.get("counts", {}).get("admitted_rows"), "code rows")
    bridge_bytes = _positive(
        bridge.get("train", {}).get("text_utf8_bytes"), "connection bytes"
    )
    bridge_rows = _positive(
        bridge.get("counts", {}).get("train_documents"), "connection rows"
    )
    development_rows = _positive(
        bridge.get("counts", {}).get("development_documents_excluded"),
        "connection development rows",
    )

    if (
        foundation.get("status") != "complete_practical_training_corpus_readiness_audit"
        or foundation.get("practical_training_corpus_ready") is not True
        or foundation.get("training_ready") is not True
        or foundation.get("bounds", {}).get("combined_byte_bound_satisfied") is not True
        or foundation.get("quality", {}).get("english_only") is not True
        or code.get("status") != "complete_common_pile_stack_edu_practical_admission"
        or code.get("practical_pretraining_ready") is not True
        or code.get("training_ready") is not True
        or code_publication.get("status")
        != "complete_stack_edu_practical_hf_metadata_publication"
        or code_publication.get("admission_receipt_sha256") != code["receipt_sha256"]
        or code_publication.get("source_text_uploaded") is not False
        or code_publication.get("training_ready") is not True
        or bridge.get("status") != "complete_bridge_training_component_admission"
        or bridge.get("connection_component_admission_authorized") is not True
        or bridge.get("transfer_ablation_complete") is not True
        or bridge.get("development_rows_physically_excluded") is not True
        or bridge.get("training_ready") is not True
        or bridge_publication.get("status")
        != "complete_bridge_training_component_hf_publication"
        or bridge_publication.get("admission_receipt_sha256")
        != bridge["receipt_sha256"]
        or bridge_publication.get("train_documents") != bridge_rows
        or bridge_publication.get("train_text_utf8_bytes") != bridge_bytes
        or bridge_publication.get("development_rows_uploaded") is not False
        or bridge_publication.get("transfer_ablation_complete") is not True
        or bridge_publication.get("training_ready") is not True
        or any(
            payload.get("four_b_training_authorized") is not False
            for payload in (foundation, code, bridge, bridge_publication)
        )
    ):
        raise FinalTrainingReleaseError("final component evidence differs")

    components = [
        {
            "component": "english_reality_anchor_foundation",
            "receipt_sha256": foundation["receipt_sha256"],
            "rows": foundation_rows,
            "logical_text_utf8_bytes": foundation_bytes,
            "custody": "private_books_plus_pinned_public_locators",
        },
        {
            "component": "educational_code_overlay",
            "receipt_sha256": code["receipt_sha256"],
            "publication_receipt_sha256": code_publication["receipt_sha256"],
            "rows": code_rows,
            "logical_text_utf8_bytes": code_bytes,
            "custody": "pinned_public_locators",
        },
        {
            "component": "verified_cross_domain_connections",
            "receipt_sha256": bridge["receipt_sha256"],
            "publication_receipt_sha256": bridge_publication["receipt_sha256"],
            "rows": bridge_rows,
            "logical_text_utf8_bytes": bridge_bytes,
            "development_rows_physically_excluded": development_rows,
            "custody": "published_train_only_gzip",
        },
    ]
    payload = {
        "schema": SCHEMA,
        "status": "complete_sai_training_data_release",
        "inputs": {
            "foundation_file_sha256": sha256_file(foundation_path),
            "code_admission_file_sha256": sha256_file(code_admission_path),
            "code_publication_file_sha256": sha256_file(code_publication_path),
            "bridge_admission_file_sha256": sha256_file(bridge_admission_path),
            "bridge_publication_file_sha256": sha256_file(bridge_publication_path),
        },
        "components": components,
        "ordered_components_sha256": canonical_sha256(components),
        "totals": {
            "components": len(components),
            "rows": foundation_rows + code_rows + bridge_rows,
            "logical_text_utf8_bytes": foundation_bytes + code_bytes + bridge_bytes,
            "foundation_text_utf8_bytes": foundation_bytes,
            "overlay_text_utf8_bytes": code_bytes + bridge_bytes,
        },
        "foundation_within_1_9_to_2_0_trillion_bytes": (
            1_900_000_000_000 <= foundation_bytes <= 2_000_000_000_000
        ),
        "english_non_slop_foundation_complete": True,
        "educational_code_overlay_complete": True,
        "verified_cross_domain_connection_overlay_complete": True,
        "cross_domain_connection_transfer_ablation_complete": True,
        "connection_development_rows_physically_excluded": True,
        "all_required_components_present": True,
        "training_data_ready": True,
        "model_training_started": False,
        "four_b_training_authorized": False,
    }
    if payload["foundation_within_1_9_to_2_0_trillion_bytes"] is not True:
        raise FinalTrainingReleaseError("foundation byte bound differs")
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-audit", type=Path, required=True)
    parser.add_argument("--code-admission", type=Path, required=True)
    parser.add_argument("--code-publication", type=Path, required=True)
    parser.add_argument("--bridge-admission", type=Path, required=True)
    parser.add_argument("--bridge-publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_release(
        args.foundation_audit,
        args.code_admission,
        args.code_publication,
        args.bridge_admission,
        args.bridge_publication,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

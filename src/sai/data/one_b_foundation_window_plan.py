"""Plan Sai's storage-bounded, exact 102.4B-token foundation window."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_spiral_contract import BANDS, SEQUENCE_LENGTH, _allocate
from sai.data.one_b_unique_token_ledger import SCHEMA as LEDGER_SCHEMA
from sai.data.token_stream import canonical_sha256
from sai.tokenizer.production_qualification import SCHEMA as TOKENIZER_SCHEMA

SCHEMA = "sai-1b-foundation-window-plan-v1"
WINDOW_SEQUENCES = 25_000_000
WINDOW_WEIGHTS = (65, 25, 8, 2)
SELECTION_SAFETY_PPM = 50_000


class OneBFoundationWindowPlanError(RuntimeError):
    """The token ledger, tokenizer fertility, or window allocation differs."""


def _load(path: Path, schema: str, hash_field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBFoundationWindowPlanError("foundation input differs") from error
    unsigned = {key: item for key, item in value.items() if key != hash_field}
    if (
        not path.is_file()
        or path.is_symlink()
        or value.get("schema") != schema
        or value.get(hash_field) != canonical_sha256(unsigned)
    ):
        raise OneBFoundationWindowPlanError("foundation input differs")
    return value


def build(ledger_path: Path, tokenizer_path: Path) -> dict[str, Any]:
    """Convert exact source estimates into conservative deterministic hash gates."""

    ledger = _load(ledger_path, LEDGER_SCHEMA, "receipt_sha256")
    tokenizer = _load(tokenizer_path, TOKENIZER_SCHEMA, "receipt_sha256")
    if tokenizer.get("status") != "qualified_production_48k":
        raise OneBFoundationWindowPlanError("production tokenizer is not qualified")
    bytes_per_token = tokenizer.get("corpus", {}).get("utf8_bytes_per_token")
    if not isinstance(bytes_per_token, float) or not 1.0 < bytes_per_token < 10.0:
        raise OneBFoundationWindowPlanError("tokenizer fertility differs")
    band_sequences = _allocate(WINDOW_SEQUENCES, WINDOW_WEIGHTS)
    bands = {}
    for band, sequences in zip(BANDS, band_sequences, strict=True):
        target_tokens = sequences * SEQUENCE_LENGTH
        train_bytes = ledger["counts"][f"band::{band}::split::train::text_utf8_bytes"]
        estimated_48k_tokens = max(1, round(train_bytes / bytes_per_token))
        raw_ppm = math.ceil(
            target_tokens * (1_000_000 + SELECTION_SAFETY_PPM) / estimated_48k_tokens
        )
        bands[band] = {
            "target_sequences": sequences,
            "target_tokens": target_tokens,
            "train_text_utf8_bytes": train_bytes,
            "estimated_unique_48k_tokens": estimated_48k_tokens,
            "selection_ppm": min(1_000_000, raw_ppm),
            "estimated_repetition_required": estimated_48k_tokens < target_tokens,
        }
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_1b_foundation_window_plan",
        "window_sequences": WINDOW_SEQUENCES,
        "window_tokens": WINDOW_SEQUENCES * SEQUENCE_LENGTH,
        "window_utf16_bytes": WINDOW_SEQUENCES * SEQUENCE_LENGTH * 2,
        "weights": dict(zip(BANDS, WINDOW_WEIGHTS, strict=True)),
        "bands": bands,
        "source_token_ledger_receipt_sha256": ledger["receipt_sha256"],
        "tokenizer_qualification_receipt_sha256": tokenizer["receipt_sha256"],
        "tokenizer_identity_sha256": tokenizer["tokenizer_identity_sha256"],
        "selection_function": "priority-sha256-first16-mod-1m-below-band-ppm",
        "selection_safety_ppm": SELECTION_SAFETY_PPM,
        "connection_documents_handled_separately": True,
        "maximum_connection_document_exposures": 16,
        "exact_final_trim_required": True,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise OneBFoundationWindowPlanError("foundation plan output exists")
    value = build(args.ledger, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

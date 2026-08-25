"""Seal Sai 1B launch readiness while preserving explicit training authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.final_training_release import SCHEMA as RELEASE_SCHEMA
from sai.data.one_b_foundation_aggregate import SCHEMA as WINDOW_SCHEMA
from sai.data.one_b_foundation_window_plan import SCHEMA as PLAN_SCHEMA
from sai.data.one_b_hf_publish import SCHEMA as HF_SCHEMA
from sai.data.one_b_stage_schedule import SCHEMA as SCHEDULE_SCHEMA
from sai.data.one_b_unique_token_ledger import SCHEMA as LEDGER_SCHEMA
from sai.data.token_stream import canonical_sha256
from sai.tokenizer.production_qualification import SCHEMA as TOKENIZER_SCHEMA
from sai.training.one_b_environment_receipt import SCHEMA as ENVIRONMENT_SCHEMA
from sai.training.one_b_gpu_preflight import SCHEMA as PREFLIGHT_SCHEMA
from sai.training.one_b_olmo_config import SCHEMA as CONFIG_SCHEMA
from sai.training.one_b_production_contract import build_contract

SCHEMA = "sai-1b-training-readiness-v1"


class OneBReadinessError(RuntimeError):
    """A signed input, lineage edge, measurement, or no-training gate differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBReadinessError("signed readiness input differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBReadinessError("signed readiness input differs")
    return value


def _never_trained(*payloads: dict[str, Any]) -> None:
    if any(
        payload.get("model_training_started") is not False
        or payload.get("one_b_training_authorized", False) is not False
        for payload in payloads
    ):
        raise OneBReadinessError("training authorization boundary differs")


def build(
    release_path: Path,
    ledger_path: Path,
    plan_path: Path,
    window_path: Path,
    schedule_path: Path,
    tokenizer_path: Path,
    config_path: Path,
    environment_path: Path,
    preflight_path: Path,
    hf_path: Path,
) -> dict[str, Any]:
    """Recompute the complete start-readiness gate without launching training."""

    release = _load_signed(release_path, RELEASE_SCHEMA)
    ledger = _load_signed(ledger_path, LEDGER_SCHEMA)
    plan = _load_signed(plan_path, PLAN_SCHEMA)
    window = _load_signed(window_path, WINDOW_SCHEMA)
    schedule = _load_signed(schedule_path, SCHEDULE_SCHEMA)
    tokenizer = _load_signed(tokenizer_path, TOKENIZER_SCHEMA)
    config = _load_signed(config_path, CONFIG_SCHEMA)
    environment = _load_signed(environment_path, ENVIRONMENT_SCHEMA)
    preflight = _load_signed(preflight_path, PREFLIGHT_SCHEMA)
    hf = _load_signed(hf_path, HF_SCHEMA)
    _never_trained(
        ledger,
        plan,
        window,
        schedule,
        tokenizer,
        config,
        environment,
        preflight,
        hf,
    )
    contract = build_contract()
    train_rows = ledger.get("counts", {}).get("split::train::rows", 0)
    development_rows = ledger.get("counts", {}).get("split::development::rows", 0)
    band_rows = {
        band: ledger.get("counts", {}).get(f"band::{band}::split::train::rows", 0)
        for band in ("foundation", "intermediate", "advanced", "expert")
    }
    if (
        release.get("status") != "complete_sai_training_data_release"
        or release.get("training_data_ready") is not True
        or release.get("connection_development_rows_physically_excluded") is not True
        or release.get("totals", {}).get("rows") != train_rows + development_rows
        or not 1_900_000_000_000
        <= release.get("totals", {}).get("foundation_text_utf8_bytes", 0)
        <= 2_000_000_000_000
        or train_rows <= 0
        or development_rows <= 0
        or any(rows <= 0 for rows in band_rows.values())
        or plan.get("source_token_ledger_receipt_sha256") != ledger["receipt_sha256"]
        or plan.get("tokenizer_qualification_receipt_sha256")
        != tokenizer["receipt_sha256"]
        or window.get("plan_receipt_sha256") != plan["receipt_sha256"]
        or window.get("tokenizer_identity_sha256")
        != tokenizer.get("tokenizer_identity_sha256")
        or schedule.get("foundation_window_receipt_sha256") != window["receipt_sha256"]
        or schedule.get("tokenizer_identity_sha256")
        != tokenizer.get("tokenizer_identity_sha256")
        or schedule.get("total_tokens") != contract["target_tokens"]
        or schedule.get("development_rows_excluded") is not True
        or tokenizer.get("status") != "qualified_production_48k"
        or tokenizer.get("byte_fallback") is not True
        or config.get("schedule_receipt_sha256") != schedule["receipt_sha256"]
        or config.get("tokenizer_qualification_receipt_sha256")
        != tokenizer["receipt_sha256"]
        or config.get("production_contract_receipt_sha256")
        != contract["receipt_sha256"]
        or config.get("exact_body_and_boundary_batches") is not True
        or config.get("document_boundary_isolation_enabled") is not True
        or environment.get("status") != "complete_nontraining_1b_olmo_environment"
        or preflight.get("status") != "passed_nontraining_h100_forward_backward_no_step"
        or preflight.get("environment_receipt_sha256") != environment["receipt_sha256"]
        or preflight.get("config_bundle_receipt_sha256") != config["receipt_sha256"]
        or preflight.get("parameter_count") != contract["parameter_ledger"]["total"]
        or preflight.get("optimizer_constructed") is not False
        or preflight.get("optimizer_update_performed") is not False
        or preflight.get("checkpoint_written") is not False
        or hf.get("status") != "complete_self_contained_1b_packed_hf_publication"
        or hf.get("schedule_receipt_sha256") != schedule["receipt_sha256"]
        or hf.get("tokenizer_qualification_receipt_sha256")
        != tokenizer["receipt_sha256"]
        or hf.get("config_receipt_sha256") != config["receipt_sha256"]
        or hf.get("all_packed_lfs_identities_verified") is not True
        or hf.get("all_remote_identities_verified") is not True
        or hf.get("development_rows_excluded") is not True
        or hf.get("packed_training_tokens_uploaded") is not True
        or hf.get("directly_trainable_after_download") is not True
        or hf.get("physical_data_files", 0) <= 0
        or hf.get("physical_data_bytes", 0) <= 0
    ):
        raise OneBReadinessError("one billion parameter readiness gate differs")
    payload = {
        "schema": SCHEMA,
        "status": "training_ready_awaiting_explicit_user_order",
        "source_release_receipt_sha256": release["receipt_sha256"],
        "source_rows": release["totals"]["rows"],
        "source_logical_text_utf8_bytes": release["totals"]["logical_text_utf8_bytes"],
        "curriculum_ledger_receipt_sha256": ledger["receipt_sha256"],
        "curriculum_train_rows": train_rows,
        "curriculum_development_rows_excluded": development_rows,
        "curriculum_train_band_rows": band_rows,
        "tokenizer_qualification_receipt_sha256": tokenizer["receipt_sha256"],
        "tokenizer_identity_sha256": tokenizer["tokenizer_identity_sha256"],
        "physical_window_receipt_sha256": window["receipt_sha256"],
        "physical_window_tokens": window["window_tokens"],
        "schedule_receipt_sha256": schedule["receipt_sha256"],
        "scheduled_tokens": schedule["total_tokens"],
        "config_receipt_sha256": config["receipt_sha256"],
        "environment_receipt_sha256": environment["receipt_sha256"],
        "h100_preflight_receipt_sha256": preflight["receipt_sha256"],
        "single_h100_steady_tokens_per_second": preflight["measurement"][
            "steady_tokens_per_second_single_h100"
        ],
        "parameter_count": preflight["parameter_count"],
        "hf_publication_receipt_sha256": hf["receipt_sha256"],
        "hf_repository": hf["repository"],
        "hf_revision": hf["revision"],
        "hf_physical_data_bytes": hf["physical_data_bytes"],
        "all_required_launch_inputs_hash_bound": True,
        "training_command_executed": False,
        "optimizer_update_performed": False,
        "model_training_started": False,
        "one_b_training_authorized": False,
        "explicit_user_order_required_to_launch": True,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--hf-publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise OneBReadinessError("readiness output already exists")
    value = build(
        args.release,
        args.ledger,
        args.plan,
        args.window,
        args.schedule,
        args.tokenizer,
        args.config,
        args.environment,
        args.preflight,
        args.hf_publication,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(args.output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

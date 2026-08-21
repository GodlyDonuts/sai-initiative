"""Validate the prospective Sai architecture tournament without authorizing training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "sai-frontier-architecture-tournament-v1"
SCALE_CONTRACT = [
    ("mechanics", 0, "cpu_and_kernel_qualification", 0, "100m"),
    ("100m", 100_000_000, "stability_efficiency_and_convergence_screen", 3, "300m"),
    ("300m", 300_000_000, "single_factor_capability_screen", 3, "1b"),
    ("1b", 1_000_000_000, "survivor_confirmation_and_interactions", 3, "4b"),
    ("4b", 4_000_000_000, "selected_stack_and_public_confirmation", 1, None),
]
CORE_MIXERS = {
    "gated_gqa",
    "three_gated_deltanet_one_gated_attention",
    "three_kda_one_gated_mla",
}
PRIMARY_TOKENIZER_SIZES = [64_000, 48_000, 32_000]
DIAGNOSTIC_TOKENIZER_SIZES = [16_000]
PROTECTED_TOKENIZER_CAPABILITIES = {
    "special_tokens",
    "byte_fallback",
    "ascii",
    "english",
    "code",
    "whitespace_and_indentation",
    "identifiers",
    "urls",
    "numbers_and_units",
    "greek",
    "math",
    "latex",
    "science_notation",
}
SECONDARY_FACTORS = [
    "partial_rope_vs_nope_mla",
    "swiglu_vs_situ_glu",
    "adamw_vs_muon_hybrid",
    "ntp_vs_ntp_plus_mtp",
    "none_vs_engram",
    "prenorm_vs_block_attnres",
    "future_summary_prediction_exploratory",
]
REQUIRED_MATCHING = {
    "one_changed_factor_per_primary_contrast": True,
    "same_data_order": True,
    "same_sequence_curriculum": True,
    "same_decoding": True,
    "primary_contrasts": {
        "iso_data": {
            "same_admitted_utf8_bytes": True,
            "model_flops_may_differ_and_are_reported": True,
        },
        "iso_flop": {
            "same_model_flops": True,
            "same_ordered_stream_prefix": True,
            "admitted_utf8_bytes_may_differ_and_are_reported": True,
        },
    },
    "cross_tokenizer_loss_unit": "negative_log_likelihood_per_utf8_byte",
    "promotion_requires_positive_paired_95ci": True,
    "material_domain_regression_allowed": False,
}
FOUR_B_PREREQUISITES = {
    "mechanics_passed",
    "100m_passed",
    "300m_passed",
    "1b_passed",
    "winning_stack_frozen",
    "official_training_order_received",
}
REQUIRED_EXCLUSIONS = {
    "always_revise",
    "mandatory_long_reasoning",
    "moe_for_4b_total_target",
    "attnres_and_mhc_combined_initially",
    "unverified_deepseek_v4_sparse_attention",
    "fusedkv_on_kda_mla_branch",
    "architecture_promotion_from_perplexity_alone",
}
REQUIRED_SOURCES = {
    "https://arxiv.org/abs/2510.26692",
    "https://github.com/MoonshotAI/Kimi-K3",
    "https://arxiv.org/abs/2505.06708",
    "https://arxiv.org/abs/2603.15031",
    "https://github.com/deepseek-ai/Engram",
    "https://arxiv.org/abs/2404.19737",
    "https://arxiv.org/abs/2510.14751",
}


class TournamentError(RuntimeError):
    """The prospective tournament is incomplete or permits premature training."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_target(target: Any) -> None:
    if not isinstance(target, dict):
        raise TournamentError("target contract is missing")
    if (
        target.get("parameter_class") != "approximately_4b_total"
        or target.get("parameterization") != "dense"
        or target.get("modality") != "text_only"
        or target.get("unicode_roundtrip") != "lossless_byte_fallback"
        or target.get("tied_embeddings") is not True
        or target.get("primary_domains")
        != ["english", "code", "math", "science", "technical"]
    ):
        raise TournamentError("fixed Sai target differs")


def _validate_scales(scales: Any) -> None:
    if not isinstance(scales, list) or len(scales) != len(SCALE_CONTRACT):
        raise TournamentError("exact scale ladder is required")
    for stage, expected in zip(scales, SCALE_CONTRACT, strict=True):
        if not isinstance(stage, dict):
            raise TournamentError("scale stage must be an object")
        observed = (
            stage.get("name"),
            stage.get("approximate_parameters"),
            stage.get("purpose"),
            stage.get("minimum_seeds"),
            stage.get("may_promote_to"),
        )
        if observed != expected:
            raise TournamentError(f"scale stage differs: {expected[0]}")


def _validate_tokenizers(tokenizers: Any) -> None:
    if not isinstance(tokenizers, dict):
        raise TournamentError("tokenizer tournament is missing")
    protected = tokenizers.get("protected_capabilities")
    if (
        tokenizers.get("primary_vocab_sizes") != PRIMARY_TOKENIZER_SIZES
        or tokenizers.get("diagnostic_only_vocab_sizes") != DIAGNOSTIC_TOKENIZER_SIZES
        or tokenizers.get("separate_tokenizer_and_reallocation_effects") is not True
        or not isinstance(protected, list)
        or len(protected) != len(set(protected))
        or set(protected) != PROTECTED_TOKENIZER_CAPABILITIES
    ):
        raise TournamentError("tokenizer candidates or protections differ")


def validate(payload: Any) -> dict[str, Any]:
    """Return a plan receipt while retaining the explicit no-training hold."""

    if not isinstance(payload, dict):
        raise TournamentError("architecture plan must be an object")
    if payload.get("schema") != SCHEMA or payload.get("status") != "prospective":
        raise TournamentError("architecture plan schema/status differs")
    if (
        payload.get("training_hold") is not True
        or payload.get("training_authorized") is not False
        or payload.get("official_training_order_received") is not False
        or payload.get("gpu_jobs_submitted") != 0
        or payload.get("training_updates_completed") != 0
    ):
        raise TournamentError("no-training boundary differs")

    _validate_target(payload.get("target"))
    _validate_scales(payload.get("scales"))
    if payload.get("matching") != REQUIRED_MATCHING:
        raise TournamentError("matched-comparison contract differs")
    core_mixers = payload.get("core_mixer_candidates")
    if (
        not isinstance(core_mixers, list)
        or len(core_mixers) != len(set(core_mixers))
        or set(core_mixers) != CORE_MIXERS
    ):
        raise TournamentError("core mixer tournament differs")
    _validate_tokenizers(payload.get("tokenizer_candidates"))
    if payload.get("ordered_secondary_factors") != SECONDARY_FACTORS:
        raise TournamentError("secondary factor order differs")

    prerequisites = payload.get("four_b_prerequisites")
    if (
        not isinstance(prerequisites, dict)
        or set(prerequisites) != FOUR_B_PREREQUISITES
        or any(value is not False for value in prerequisites.values())
    ):
        raise TournamentError("4B prerequisites must remain unmet before training")
    exclusions = payload.get("explicit_exclusions")
    if (
        not isinstance(exclusions, list)
        or len(exclusions) != len(set(exclusions))
        or set(exclusions) != REQUIRED_EXCLUSIONS
    ):
        raise TournamentError("explicit architecture exclusions differ")
    sources = payload.get("primary_sources")
    if (
        not isinstance(sources, list)
        or len(sources) != len(set(sources))
        or set(sources) != REQUIRED_SOURCES
    ):
        raise TournamentError("primary research sources differ")

    return {
        "schema": "sai-frontier-architecture-tournament-receipt-v1",
        "status": "prospective_plan_validated",
        "training_authorized": False,
        "official_training_order_required": True,
        "plan_sha256": canonical_sha256(payload),
        "scale_order": [stage[0] for stage in SCALE_CONTRACT],
        "core_mixer_candidates": sorted(CORE_MIXERS),
        "primary_tokenizer_sizes": PRIMARY_TOKENIZER_SIZES,
    }


def load_and_validate(path: Path) -> dict[str, Any]:
    return validate(json.loads(path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(load_and_validate(args.plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a deterministic prerequisite-to-specialization Sai curriculum.

The curriculum is deliberately model independent.  It does not pretend that a
surface heuristic is a semantic prerequisite graph; instead it binds measurable
document complexity, removes residual junk and high-confidence near duplicates,
and paces four immutable difficulty bands with foundation rehearsal in every
later phase.  The output remains ordinary benchmark-disjoint
``sai-pretraining-document-v1`` JSONL so the token-stream freezer can replay it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import unicodedata
import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from sai.data.decontamination import RECEIPT_SCHEMA as DECONTAMINATION_SCHEMA
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-curriculum-order-receipt-v1"
BANDS = ("foundation", "composition", "reasoning", "specialization")
PHASES = ("grounding", "integration", "reasoning", "specialization")
BUCKETS_PER_BAND = 16
SHINGLE_WIDTH = 5
SKETCH_BUCKETS = 8
NEAR_DUPLICATE_MATCHES = 6
MAX_SKETCH_SHINGLES = 128
_U256_MODULUS = 1 << 256

# Fractions of each difficulty band's own population released in each phase.
# Columns sum to one.  Foundation rehearsal therefore remains present after the
# grounding phase while specialized material is delayed rather than front-loaded.
BAND_PHASE_SHARES: dict[str, tuple[float, float, float, float]] = {
    "foundation": (0.50, 0.25, 0.15, 0.10),
    "composition": (0.20, 0.35, 0.30, 0.15),
    "reasoning": (0.05, 0.20, 0.40, 0.35),
    "specialization": (0.00, 0.05, 0.30, 0.65),
}

_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_SENTENCE = re.compile(r"[.!?]+(?:\s+|$)")
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_CODE_LINE = re.compile(
    r"^\s*(?:def |class |function |import |from |#include|SELECT |INSERT |"
    r"if\s*\(|for\s*\(|while\s*\(|[{}]|```)",
    re.IGNORECASE,
)
_REASONING = re.compile(
    r"\b(?:because|therefore|however|consequently|implies|assume|suppose|"
    r"derive|proof|hypothesis|evidence|if and only if|for example|in contrast)\b",
    re.IGNORECASE,
)
_TECHNICAL = re.compile(
    r"\b(?:algorithm|function|variable|equation|theorem|molecule|protein|"
    r"quantum|compiler|database|probability|derivative|integral|vector|matrix|"
    r"network|protocol|experiment|coefficient|simulation|architecture)\b",
    re.IGNORECASE,
)


class CurriculumError(RuntimeError):
    """The source, quality evidence, ordering, or curriculum receipt differs."""


def _ratio(value: float, low: float, high: float) -> float:
    if high <= low:
        raise CurriculumError("curriculum normalization geometry differs")
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.casefold())
    count = max(1, len(groups))
    if len(word) > 3 and word.casefold().endswith("e") and count > 1:
        count -= 1
    return count


def document_signals(text: str) -> dict[str, Any]:
    """Return deterministic quality and difficulty evidence for one document."""

    if not isinstance(text, str) or not text:
        raise CurriculumError("curriculum document text differs")
    words = _WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    characters = len(text)
    alphabetic = [character for character in text if character.isalpha()]
    latin = sum("LATIN" in unicodedata.name(character, "") for character in alphabetic)
    unique_words = len(set(words))
    sentences = max(1, len(_SENTENCE.findall(text)))
    average_sentence_words = len(words) / sentences
    average_word_characters = (
        sum(len(word) for word in words) / len(words) if words else 0.0
    )
    long_word_fraction = (
        sum(len(word) >= 9 for word in words) / len(words) if words else 0.0
    )
    lexical_diversity = unique_words / math.sqrt(max(1, len(words)))
    symbol_fraction = (
        sum(not character.isalnum() and not character.isspace() for character in text)
        / characters
    )
    digit_fraction = sum(character.isdigit() for character in text) / characters
    lines = [line for line in text.splitlines() if line.strip()]
    code_line_fraction = (
        sum(bool(_CODE_LINE.search(line)) for line in lines) / len(lines)
        if lines
        else 0.0
    )
    reasoning_density = len(_REASONING.findall(text)) / max(1, len(words))
    technical_density = len(_TECHNICAL.findall(text)) / max(1, len(words))
    syllables = sum(_syllables(word) for word in words)
    reading_ease = (
        206.835 - 1.015 * average_sentence_words - 84.6 * syllables / max(1, len(words))
    )
    latin_fraction = latin / len(alphabetic) if alphabetic else 0.0
    unique_word_fraction = unique_words / max(1, len(words))
    longest_character_run = max(
        (len(match.group(0)) for match in re.finditer(r"(.)\1+", text)),
        default=1,
    )
    url_density = len(_URL.findall(text)) / max(1, len(words))

    quality_reasons = []
    if len(words) < 80:
        quality_reasons.append("too_few_words")
    if len(alphabetic) >= 100 and latin_fraction < 0.80:
        quality_reasons.append("latin_coverage_low")
    if unique_word_fraction < 0.10:
        quality_reasons.append("lexical_repetition_high")
    if longest_character_run > 40:
        quality_reasons.append("character_run_high")
    if url_density > 0.02:
        quality_reasons.append("url_density_high")

    difficulty = (
        0.21 * _ratio(average_sentence_words, 8.0, 32.0)
        + 0.15 * _ratio(average_word_characters, 4.0, 7.0)
        + 0.13 * _ratio(long_word_fraction, 0.04, 0.28)
        + 0.08 * _ratio(lexical_diversity, 4.0, 16.0)
        + 0.08 * _ratio(symbol_fraction, 0.015, 0.16)
        + 0.07 * _ratio(digit_fraction, 0.005, 0.12)
        + 0.09 * _ratio(code_line_fraction, 0.0, 0.35)
        + 0.08 * _ratio(reasoning_density, 0.0, 0.025)
        + 0.07 * _ratio(technical_density, 0.0, 0.035)
        + 0.04 * _ratio(math.log10(max(400, characters)), 2.6, 5.0)
    )
    # Reading ease is reported independently and provides a monotonic sanity
    # check without being counted twice in the frozen composite.
    difficulty = min(1.0, max(0.0, difficulty))
    if difficulty < 0.13:
        band = "foundation"
    elif difficulty < 0.22:
        band = "composition"
    elif difficulty < 0.42:
        band = "reasoning"
    else:
        band = "specialization"
    return {
        "words": len(words),
        "sentences": sentences,
        "average_sentence_words": average_sentence_words,
        "average_word_characters": average_word_characters,
        "long_word_fraction": long_word_fraction,
        "lexical_diversity_sqrt_normalized": lexical_diversity,
        "unique_word_fraction": unique_word_fraction,
        "symbol_fraction": symbol_fraction,
        "digit_fraction": digit_fraction,
        "code_line_fraction": code_line_fraction,
        "reasoning_marker_density": reasoning_density,
        "technical_marker_density": technical_density,
        "reading_ease": reading_ease,
        "latin_alphabet_fraction": latin_fraction,
        "longest_character_run": longest_character_run,
        "url_density": url_density,
        "quality_reasons": quality_reasons,
        "quality_accepted": not quality_reasons,
        "difficulty": difficulty,
        "band": band,
    }


def _near_duplicate_sketch(text: str) -> tuple[int, ...]:
    words = _WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    if len(words) < SHINGLE_WIDTH:
        return tuple()
    minima = [(1 << 64) - 1] * SKETCH_BUCKETS
    available = len(words) - SHINGLE_WIDTH + 1
    if available <= MAX_SKETCH_SHINGLES:
        positions = range(available)
    else:
        positions = (
            index * (available - 1) // (MAX_SKETCH_SHINGLES - 1)
            for index in range(MAX_SKETCH_SHINGLES)
        )
    for index in positions:
        shingle = "\x1f".join(words[index : index + SHINGLE_WIDTH]).encode()
        digest = int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        bucket = digest & (SKETCH_BUCKETS - 1)
        if digest < minima[bucket]:
            minima[bucket] = digest
    return tuple(minima)


def _score_candidate(
    line: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, tuple[int, ...] | None]:
    """Compute expensive row-local evidence; safe for ordered process pools."""

    try:
        row = normalize_document(json.loads(line))
    except (json.JSONDecodeError, RuntimeError):
        return None, None, None
    return row, document_signals(row["text"]), _near_duplicate_sketch(row["text"])


def _is_near_duplicate(
    sketch: tuple[int, ...],
    accepted: list[tuple[int, ...]],
    index: dict[tuple[int, int, int], set[int]],
) -> bool:
    if len(sketch) != SKETCH_BUCKETS:
        return False
    candidates: set[int] = set()
    for pair in range(0, SKETCH_BUCKETS, 2):
        candidates.update(index.get((pair // 2, sketch[pair], sketch[pair + 1]), ()))
    for candidate in candidates:
        previous = accepted[candidate]
        if sum(left == right for left, right in zip(sketch, previous, strict=True)) >= (
            NEAR_DUPLICATE_MATCHES
        ):
            return True
    return False


def _add_sketch(
    sketch: tuple[int, ...],
    accepted: list[tuple[int, ...]],
    index: dict[tuple[int, int, int], set[int]],
) -> None:
    if len(sketch) != SKETCH_BUCKETS:
        return
    identity = len(accepted)
    accepted.append(sketch)
    for pair in range(0, SKETCH_BUCKETS, 2):
        index.setdefault((pair // 2, sketch[pair], sketch[pair + 1]), set()).add(
            identity
        )


def _largest_remainder(count: int, shares: tuple[float, ...]) -> list[int]:
    exact = [count * share for share in shares]
    result = [math.floor(value) for value in exact]
    remaining = count - sum(result)
    order = sorted(
        range(len(shares)), key=lambda index: (-(exact[index] - result[index]), index)
    )
    for index in order[:remaining]:
        result[index] += 1
    return result


def _identity_fingerprint(
    *, count: int, xor_value: int, sum_value: int
) -> dict[str, Any]:
    """Return an order-independent exact-identity multiset fingerprint."""

    return {
        "count": count,
        "xor_u256": f"{xor_value:064x}",
        "sum_mod_u256": f"{sum_value % _U256_MODULUS:064x}",
    }


def _bucket_order(band: str) -> list[int]:
    return sorted(
        range(BUCKETS_PER_BAND),
        key=lambda bucket: hashlib.sha256(
            f"sai-curriculum-v1:{band}:{bucket}".encode()
        ).digest(),
    )


def _band_iterator(stage: Path, band: str) -> Iterator[dict[str, Any]]:
    for bucket in _bucket_order(band):
        path = stage / f"{band}.{bucket:02d}.jsonl"
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                yield json.loads(line)


def _validate_decontamination_receipt(source: Path, receipt: Path) -> dict[str, Any]:
    if not receipt.is_file() or receipt.is_symlink():
        raise CurriculumError("decontamination receipt is missing or unsafe")
    payload = json.loads(receipt.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != DECONTAMINATION_SCHEMA:
        raise CurriculumError("decontamination receipt schema differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise CurriculumError("decontamination receipt hash differs")
    expected = payload.get("output")
    if (
        payload.get("status") != "passed"
        or not isinstance(expected, dict)
        or expected.get("path") != str(source.resolve())
        or expected.get("bytes") != source.stat().st_size
        or expected.get("sha256") != sha256_file(source)
    ):
        raise CurriculumError("decontaminated source differs")
    return payload


def build_curriculum(
    source: Path,
    decontamination_receipt: Path,
    output: Path,
    receipt: Path,
    *,
    minimum_documents_per_band: int = 100,
    workers: int = 1,
) -> dict[str, Any]:
    """Filter, deduplicate, stratify, and deterministically pace one corpus."""

    if not source.is_file() or source.is_symlink():
        raise CurriculumError("curriculum source is missing or unsafe")
    if any(path.exists() or path.is_symlink() for path in (output, receipt)):
        raise CurriculumError("curriculum output already exists")
    if (
        isinstance(minimum_documents_per_band, bool)
        or not isinstance(minimum_documents_per_band, int)
        or minimum_documents_per_band <= 0
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 64
    ):
        raise CurriculumError("curriculum population or worker geometry differs")
    decontamination = _validate_decontamination_receipt(source, decontamination_receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.name}.curriculum.{uuid.uuid4().hex}"
    output_stage = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    receipt_stage = receipt.parent / f".{receipt.name}.partial.{uuid.uuid4().hex}"
    work.mkdir(mode=0o700)
    handles: dict[tuple[str, int], TextIO] = {}
    band_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    sketches: list[tuple[int, ...]] = []
    sketch_index: dict[tuple[int, int, int], set[int]] = {}
    scanned = 0
    accepted_identity_xor = 0
    accepted_identity_sum = 0
    pool = None
    try:
        with source.open() as source_handle:
            lines = (line for line in source_handle if line.strip())
            if workers == 1:
                candidates = map(_score_candidate, lines)
            else:
                if os.name != "posix":
                    raise CurriculumError(
                        "parallel curriculum scoring requires a POSIX fork runtime"
                    )
                pool = multiprocessing.get_context("fork").Pool(processes=workers)
                candidates = pool.imap(_score_candidate, lines, chunksize=64)
            for row, signals, sketch in candidates:
                scanned += 1
                if row is None or signals is None or sketch is None:
                    rejected["malformed_or_unverified"] += 1
                    continue
                if not signals["quality_accepted"]:
                    for reason in signals["quality_reasons"]:
                        rejected[reason] += 1
                    continue
                if _is_near_duplicate(sketch, sketches, sketch_index):
                    rejected["near_duplicate"] += 1
                    continue
                _add_sketch(sketch, sketches, sketch_index)
                band = signals["band"]
                bucket = int(row["identity_sha256"][:8], 16) % BUCKETS_PER_BAND
                key = (band, bucket)
                if key not in handles:
                    handles[key] = (work / f"{band}.{bucket:02d}.jsonl").open("w")
                envelope = {
                    "band": band,
                    "difficulty": signals["difficulty"],
                    "row": row,
                }
                handles[key].write(
                    json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
                )
                band_counts[band] += 1
                identity_integer = int(row["identity_sha256"], 16)
                accepted_identity_xor ^= identity_integer
                accepted_identity_sum = (
                    accepted_identity_sum + identity_integer
                ) % _U256_MODULUS
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
        for handle in handles.values():
            handle.close()
        handles.clear()
        accepted = sum(band_counts.values())
        if not scanned or not accepted:
            raise CurriculumError("curriculum admitted no documents")
        if any(band_counts[band] < minimum_documents_per_band for band in BANDS):
            raise CurriculumError(
                "curriculum difficulty band population is insufficient"
            )

        allocations = {
            band: _largest_remainder(band_counts[band], BAND_PHASE_SHARES[band])
            for band in BANDS
        }
        iterators = {band: _band_iterator(work, band) for band in BANDS}
        phase_rows: dict[str, dict[str, Any]] = {}
        emitted_identity_xor = 0
        emitted_identity_sum = 0
        with output_stage.open("w") as output_handle:
            for phase_index, phase in enumerate(PHASES):
                targets = {band: allocations[band][phase_index] for band in BANDS}
                phase_total = sum(targets.values())
                emitted = {band: 0 for band in BANDS}
                difficulty_sum = 0.0
                phase_identity = hashlib.sha256()
                for position in range(phase_total):
                    available = [
                        band for band in BANDS if emitted[band] < targets[band]
                    ]
                    if not available:
                        raise CurriculumError("curriculum phase allocation ended early")
                    band = max(
                        available,
                        key=lambda candidate: (
                            targets[candidate] * (position + 1) / max(1, phase_total)
                            - emitted[candidate],
                            -BANDS.index(candidate),
                        ),
                    )
                    try:
                        envelope = next(iterators[band])
                    except StopIteration as error:
                        raise CurriculumError(
                            "curriculum band population ended early"
                        ) from error
                    row = envelope["row"]
                    encoded = (
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    output_handle.write(encoded)
                    identity = bytes.fromhex(row["identity_sha256"])
                    phase_identity.update(identity)
                    identity_integer = int.from_bytes(identity, "big")
                    emitted_identity_xor ^= identity_integer
                    emitted_identity_sum = (
                        emitted_identity_sum + identity_integer
                    ) % _U256_MODULUS
                    difficulty_sum += float(envelope["difficulty"])
                    emitted[band] += 1
                phase_rows[phase] = {
                    "index": phase_index,
                    "documents": phase_total,
                    "by_band": emitted,
                    "mean_difficulty": difficulty_sum / max(1, phase_total),
                    "identity_sha256": phase_identity.hexdigest(),
                }
        for band in BANDS:
            try:
                next(iterators[band])
            except StopIteration:
                pass
            else:
                raise CurriculumError("curriculum band has unconsumed documents")

        means = [phase_rows[phase]["mean_difficulty"] for phase in PHASES]
        monotonic = all(
            left <= right for left, right in zip(means, means[1:], strict=False)
        )
        first = phase_rows[PHASES[0]]["by_band"]
        last = phase_rows[PHASES[-1]]["by_band"]
        progression = {
            "phase_mean_difficulty_nondecreasing": monotonic,
            "grounding_has_no_specialization": first["specialization"] == 0,
            "foundation_frontloaded": first["foundation"] > last["foundation"],
            "specialization_backloaded": last["specialization"]
            > first["specialization"],
            "foundation_rehearsed_in_final_phase": last["foundation"] > 0,
        }
        qualified = all(progression.values())
        accepted_fingerprint = _identity_fingerprint(
            count=accepted,
            xor_value=accepted_identity_xor,
            sum_value=accepted_identity_sum,
        )
        emitted_fingerprint = _identity_fingerprint(
            count=sum(row["documents"] for row in phase_rows.values()),
            xor_value=emitted_identity_xor,
            sum_value=emitted_identity_sum,
        )
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "qualified" if qualified else "failed",
            "curriculum_qualified": qualified,
            "training_authorized": False,
            "source": {
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            },
            "decontamination_receipt": {
                "path": str(decontamination_receipt.resolve()),
                "bytes": decontamination_receipt.stat().st_size,
                "sha256": sha256_file(decontamination_receipt),
                "receipt_sha256": decontamination["receipt_sha256"],
            },
            "policy": {
                "bands": list(BANDS),
                "phases": list(PHASES),
                "band_phase_shares": {
                    band: list(BAND_PHASE_SHARES[band]) for band in BANDS
                },
                "difficulty_thresholds": [0.13, 0.22, 0.42],
                "minimum_documents_per_band": minimum_documents_per_band,
                "near_duplicate": {
                    "method": "five_word_shingle_eight_bucket_lsh_minima",
                    "shingle_width": SHINGLE_WIDTH,
                    "sketch_buckets": SKETCH_BUCKETS,
                    "maximum_sampled_shingles": MAX_SKETCH_SHINGLES,
                    "matching_minima_required": NEAR_DUPLICATE_MATCHES,
                    "claim": "high_confidence_near_duplicate_filter_not_exhaustive",
                },
                "quality_rules": {
                    "minimum_words": 80,
                    "latin_alphabet_fraction_minimum": 0.80,
                    "unique_word_fraction_minimum": 0.10,
                    "longest_character_run_maximum": 40,
                    "url_density_maximum": 0.02,
                },
            },
            "documents": {
                "scanned": scanned,
                "accepted": accepted,
                "rejected": scanned - accepted,
                "rejected_by_reason": dict(sorted(rejected.items())),
                "accepted_by_band": {band: band_counts[band] for band in BANDS},
                "accepted_identity_multiset": accepted_fingerprint,
                "emitted_identity_multiset": emitted_fingerprint,
                "all_accepted_emitted_once": accepted_fingerprint
                == emitted_fingerprint,
            },
            "phases": phase_rows,
            "progression_checks": progression,
            "output": {
                "path": str(output.resolve()),
                "bytes": output_stage.stat().st_size,
                "sha256": sha256_file(output_stage),
            },
            "limitations": [
                "difficulty_is_a_deterministic_surface_proxy_not_a_semantic_oracle",
                "near_duplicate_filter_is_high_confidence_not_exhaustive",
                "curriculum_requires_learning_curve_validation_against_a_shuffled_control",
                "receipt_does_not_authorize_4b_training",
            ],
        }
        payload["curriculum_qualified"] = bool(
            payload["curriculum_qualified"]
            and payload["documents"]["all_accepted_emitted_once"]
        )
        payload["status"] = "qualified" if payload["curriculum_qualified"] else "failed"
        payload["receipt_sha256"] = canonical_sha256(payload)
        receipt_stage.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(output_stage, output)
        os.replace(receipt_stage, receipt)
        return payload
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        for handle in handles.values():
            handle.close()
        output_stage.unlink(missing_ok=True)
        receipt_stage.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def validate_curriculum(receipt: Path, *, workers: int = 1) -> dict[str, Any]:
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 64
    ):
        raise CurriculumError("curriculum validation worker geometry differs")
    if not receipt.is_file() or receipt.is_symlink():
        raise CurriculumError("curriculum receipt is missing or unsafe")
    payload = json.loads(receipt.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise CurriculumError("curriculum receipt schema differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise CurriculumError("curriculum receipt hash differs")
    output = Path(payload.get("output", {}).get("path", ""))
    if (
        not output.is_file()
        or output.is_symlink()
        or payload.get("output", {}).get("bytes") != output.stat().st_size
        or payload.get("output", {}).get("sha256") != sha256_file(output)
    ):
        raise CurriculumError("curriculum output differs")
    if (
        payload.get("status") != "qualified"
        or payload.get("curriculum_qualified") is not True
        or payload.get("training_authorized") is not False
        or payload.get("documents", {}).get("all_accepted_emitted_once") is not True
        or not all(payload.get("progression_checks", {}).values())
    ):
        raise CurriculumError("curriculum qualification differs")
    expected_policy = {
        "bands": list(BANDS),
        "phases": list(PHASES),
        "band_phase_shares": {band: list(BAND_PHASE_SHARES[band]) for band in BANDS},
        "difficulty_thresholds": [0.13, 0.22, 0.42],
        "minimum_documents_per_band": payload.get("policy", {}).get(
            "minimum_documents_per_band"
        ),
        "near_duplicate": {
            "method": "five_word_shingle_eight_bucket_lsh_minima",
            "shingle_width": SHINGLE_WIDTH,
            "sketch_buckets": SKETCH_BUCKETS,
            "maximum_sampled_shingles": MAX_SKETCH_SHINGLES,
            "matching_minima_required": NEAR_DUPLICATE_MATCHES,
            "claim": "high_confidence_near_duplicate_filter_not_exhaustive",
        },
        "quality_rules": {
            "minimum_words": 80,
            "latin_alphabet_fraction_minimum": 0.80,
            "unique_word_fraction_minimum": 0.10,
            "longest_character_run_maximum": 40,
            "url_density_maximum": 0.02,
        },
    }
    minimum = expected_policy["minimum_documents_per_band"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum <= 0
        or payload.get("policy") != expected_policy
    ):
        raise CurriculumError("curriculum policy differs")
    source = Path(payload.get("source", {}).get("path", ""))
    decontamination_receipt = Path(
        payload.get("decontamination_receipt", {}).get("path", "")
    )
    if (
        not source.is_file()
        or source.is_symlink()
        or payload.get("source", {}).get("bytes") != source.stat().st_size
        or payload.get("source", {}).get("sha256") != sha256_file(source)
        or not decontamination_receipt.is_file()
        or decontamination_receipt.is_symlink()
        or payload.get("decontamination_receipt", {}).get("bytes")
        != decontamination_receipt.stat().st_size
        or payload.get("decontamination_receipt", {}).get("sha256")
        != sha256_file(decontamination_receipt)
    ):
        raise CurriculumError("curriculum source evidence differs")
    decontamination = _validate_decontamination_receipt(source, decontamination_receipt)
    if (
        payload["decontamination_receipt"].get("receipt_sha256")
        != decontamination["receipt_sha256"]
    ):
        raise CurriculumError("curriculum decontamination identity differs")

    seen_identities: set[str] = set()
    sketches: list[tuple[int, ...]] = []
    sketch_index: dict[tuple[int, int, int], set[int]] = {}
    total_by_band: Counter[str] = Counter()
    total_xor = 0
    total_sum = 0
    total_count = 0
    replayed_phases: dict[str, dict[str, Any]] = {}
    pool = None
    try:
        with output.open() as handle:
            if workers == 1:
                candidates = map(_score_candidate, handle)
            else:
                if os.name != "posix":
                    raise CurriculumError(
                        "parallel curriculum validation requires a POSIX fork runtime"
                    )
                pool = multiprocessing.get_context("fork").Pool(processes=workers)
                candidates = pool.imap(_score_candidate, handle, chunksize=64)
            for phase_index, phase in enumerate(PHASES):
                declared = payload.get("phases", {}).get(phase)
                if (
                    not isinstance(declared, dict)
                    or declared.get("index") != phase_index
                    or isinstance(declared.get("documents"), bool)
                    or not isinstance(declared.get("documents"), int)
                    or declared["documents"] <= 0
                ):
                    raise CurriculumError("curriculum phase evidence differs")
                by_band: Counter[str] = Counter()
                difficulty_sum = 0.0
                phase_identity = hashlib.sha256()
                for _ in range(declared["documents"]):
                    try:
                        row, signals, sketch = next(candidates)
                    except StopIteration as error:
                        raise CurriculumError(
                            "curriculum output ended early"
                        ) from error
                    if row is None or signals is None or sketch is None:
                        raise CurriculumError("curriculum output row differs")
                    identity = row["identity_sha256"]
                    if identity in seen_identities:
                        raise CurriculumError("curriculum identity is duplicated")
                    seen_identities.add(identity)
                    if not signals["quality_accepted"]:
                        raise CurriculumError("curriculum output failed quality replay")
                    if _is_near_duplicate(sketch, sketches, sketch_index):
                        raise CurriculumError(
                            "curriculum near duplicate replay differs"
                        )
                    _add_sketch(sketch, sketches, sketch_index)
                    band = signals["band"]
                    by_band[band] += 1
                    total_by_band[band] += 1
                    difficulty_sum += float(signals["difficulty"])
                    identity_bytes = bytes.fromhex(identity)
                    phase_identity.update(identity_bytes)
                    identity_integer = int.from_bytes(identity_bytes, "big")
                    total_xor ^= identity_integer
                    total_sum = (total_sum + identity_integer) % _U256_MODULUS
                    total_count += 1
                replayed = {
                    "index": phase_index,
                    "documents": declared["documents"],
                    "by_band": {band: by_band[band] for band in BANDS},
                    "mean_difficulty": difficulty_sum / declared["documents"],
                    "identity_sha256": phase_identity.hexdigest(),
                }
                if replayed != declared:
                    raise CurriculumError("curriculum phase replay differs")
                replayed_phases[phase] = replayed
            try:
                next(candidates)
            except StopIteration:
                pass
            else:
                raise CurriculumError("curriculum output has undeclared rows")
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    fingerprint = _identity_fingerprint(
        count=total_count, xor_value=total_xor, sum_value=total_sum
    )
    documents = payload.get("documents", {})
    if (
        documents.get("accepted") != total_count
        or documents.get("accepted_by_band")
        != {band: total_by_band[band] for band in BANDS}
        or documents.get("emitted_identity_multiset") != fingerprint
        or documents.get("accepted_identity_multiset") != fingerprint
        or payload.get("phases") != replayed_phases
    ):
        raise CurriculumError("curriculum document replay differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--decontamination-receipt", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--minimum-documents-per-band", type=int, default=100)
    build.add_argument("--workers", type=int, default=1)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_curriculum(
            args.source,
            args.decontamination_receipt,
            args.output,
            args.receipt,
            minimum_documents_per_band=args.minimum_documents_per_band,
            workers=args.workers,
        )
    else:
        payload = validate_curriculum(args.receipt, workers=args.workers)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

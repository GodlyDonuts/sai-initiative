"""Build source-safe practical locators for the pinned Common Pile Stack-Edu.

The broad PleIAs practical path is intentionally English-only.  Stack-Edu uses
programming-language labels instead, so code must be admitted through a
separate policy that understands repository metadata, permissive licenses, and
code-specific safety signals.  This scanner retains no source text: it binds
each accepted row to one fully verified compressed parent and records the exact
content hash needed for transient reconstruction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.stack_edu_audit import ALLOWED_LICENSES, LANGUAGES
from sai.data.stack_edu_safety import _policy as safety_policy
from sai.data.stack_edu_safety import _scan_text
from sai.data.token_stream import canonical_sha256, sha256_file

SOURCE_ID = "common_pile_stackv2_edu"
SOURCE_REPOSITORY = "common-pile/stackv2_edu_filtered"
SOURCE_REVISION = "c354dbe88469a1153e97c6a63ac50591849654de"
EXPECTED_PARENTS = 95
LOCATOR_SCHEMA = "sai-common-pile-stack-edu-practical-locator-v1"
SHARD_SCHEMA = "sai-common-pile-stack-edu-practical-scan-shard-v1"
MINIMUM_TEXT_BYTES = 512
MAXIMUM_TEXT_BYTES = 1_000_000
ACCEPTED_INTEGER_SCORES = frozenset({3, 4})
_REVIEW_EXCLUSIONS = frozenset(
    {
        "personal_email_candidate",
        "jwt_like_token",
        "high_entropy_token_candidate",
        "extreme_line_length_or_minification",
        "excessive_repeated_nonempty_lines",
        "generated_code_marker",
    }
)


class StackEduPracticalScanError(RuntimeError):
    """A pinned parent, row, or practical Stack-Edu invariant differs."""


def _schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise StackEduPracticalScanError("pyarrow is required") from error
    return pa.schema(
        [
            ("schema", pa.string()),
            ("source_id", pa.string()),
            ("source_repository", pa.string()),
            ("source_revision", pa.string()),
            ("source_path", pa.string()),
            ("source_parent_sha256", pa.string()),
            ("source_row_index", pa.int64()),
            ("source_row_identity_sha256", pa.string()),
            ("blob_id", pa.string()),
            ("repo_name", pa.string()),
            ("repo_path", pa.string()),
            ("language", pa.string()),
            ("licenses", pa.list_(pa.string())),
            ("integer_score", pa.int64()),
            ("text_utf8_bytes", pa.int64()),
            ("content_sha256", pa.string()),
        ]
    )


def load_parents(
    path: Path, expected_parents: int = EXPECTED_PARENTS
) -> list[dict[str, Any]]:
    """Load the exact Stack-Edu members from the materialized source manifest."""

    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or isinstance(expected_parents, bool)
        or expected_parents < 1
    ):
        raise StackEduPracticalScanError("Stack-Edu manifest is unsafe")
    parents = []
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("source_id") != SOURCE_ID:
                    continue
                if (
                    row.get("source_repository") != SOURCE_REPOSITORY
                    or row.get("source_revision") != SOURCE_REVISION
                    or not isinstance(row.get("source_path"), str)
                    or not row["source_path"].endswith(".json.gz")
                    or isinstance(row.get("bytes"), bool)
                    or not isinstance(row.get("bytes"), int)
                    or row["bytes"] <= 0
                    or not isinstance(row.get("sha256"), str)
                    or len(row["sha256"]) != 64
                ):
                    raise StackEduPracticalScanError("Stack-Edu parent differs")
                parents.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StackEduPracticalScanError("Stack-Edu manifest differs") from error
    parents.sort(key=lambda row: row["source_path"])
    if (
        len(parents) != expected_parents
        or len({row["source_path"] for row in parents}) != len(parents)
        or len({row["sha256"] for row in parents}) != len(parents)
    ):
        raise StackEduPracticalScanError("Stack-Edu parent coverage differs")
    return parents


def _route(row: Any) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(row, dict):
        return "hold_nonobject", None
    text = row.get("text")
    metadata = row.get("metadata")
    score = row.get("int_score")
    if not isinstance(text, str) or not isinstance(metadata, dict):
        return "hold_structural", None
    if isinstance(score, bool) or score not in ACCEPTED_INTEGER_SCORES:
        return "hold_educational_score", None
    encoded = text.encode()
    if len(encoded) < MINIMUM_TEXT_BYTES:
        return "hold_too_short", None
    if len(encoded) > MAXIMUM_TEXT_BYTES:
        return "hold_too_large", None
    blob_id = row.get("id")
    licenses = metadata.get("detected_licenses")
    language = metadata.get("language")
    repo_name = metadata.get("repo_name")
    repo_path = metadata.get("path")
    if (
        not isinstance(blob_id, str)
        or len(blob_id) != 40
        or any(character not in "0123456789abcdef" for character in blob_id)
        or metadata.get("blob_id") != blob_id
        or metadata.get("src_encoding") != "UTF-8"
        or metadata.get("license_type") != "permissive"
        or metadata.get("is_generated") is not False
        or metadata.get("is_vendor") is not False
        or language not in LANGUAGES
        or not isinstance(repo_name, str)
        or not repo_name
        or not isinstance(repo_path, str)
        or not repo_path.startswith("/")
        or not isinstance(licenses, list)
        or not licenses
        or any(not isinstance(value, str) for value in licenses)
        or not set(licenses).issubset(ALLOWED_LICENSES)
    ):
        return "hold_provenance_or_rights", None
    scan = _scan_text(text, source_path=repo_path)
    if scan["reject_reasons"]:
        return "hold_high_confidence_safety", None
    review_reasons = set(scan["review_reasons"])
    if review_reasons.intersection(_REVIEW_EXCLUSIONS):
        return "hold_bounded_safety_review", None
    if language == "Python" and "python3_syntax_or_version_review" in review_reasons:
        return "hold_python_syntax", None
    content_sha256 = hashlib.sha256(encoded).hexdigest()
    return "pass_practical_code_gate", {
        "blob_id": blob_id,
        "repo_name": repo_name,
        "repo_path": repo_path,
        "language": language,
        "licenses": sorted(set(licenses)),
        "integer_score": score,
        "text_utf8_bytes": len(encoded),
        "content_sha256": content_sha256,
    }


def _download(parent: dict[str, Any], token: str, root: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise StackEduPracticalScanError("huggingface_hub is required") from error
    try:
        path = Path(
            hf_hub_download(
                repo_id=SOURCE_REPOSITORY,
                filename=parent["source_path"],
                repo_type="dataset",
                revision=SOURCE_REVISION,
                token=token,
                local_dir=root,
            )
        )
    except Exception as error:
        raise StackEduPracticalScanError("Stack-Edu download failed") from error
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != parent["bytes"]
        or sha256_file(path) != parent["sha256"]
    ):
        raise StackEduPracticalScanError("Stack-Edu parent bytes differ")
    return path


def run_shard(
    manifest_path: Path,
    output_root: Path,
    shard_index: int,
    token: str,
    *,
    expected_parents: int = EXPECTED_PARENTS,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Fully verify one compressed parent and emit text-free row locators."""

    if (
        not token
        or output_root.exists()
        or output_root.is_symlink()
        or not 0 <= shard_index < expected_parents
    ):
        raise StackEduPracticalScanError("Stack-Edu scan arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise StackEduPracticalScanError("pyarrow is required") from error
    parents = load_parents(manifest_path, expected_parents)
    parent = parents[shard_index]
    output_root.mkdir(parents=True)
    output_path = output_root / "locators.parquet"
    temporary_path = output_root / f".locators.partial.{uuid.uuid4().hex}.parquet"
    schema = _schema()
    counts: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    scores: Counter[int] = Counter()
    licenses: Counter[str] = Counter()
    selected_bytes = 0
    identities = []
    writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
    pending = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-stack-edu-practical-", dir=scratch_root
        ) as directory:
            source = _download(parent, token, Path(directory))
            try:
                with gzip.open(source, "rt", encoding="utf-8") as handle:
                    for row_index, line in enumerate(handle):
                        counts["source_rows"] += 1
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise StackEduPracticalScanError(
                                "Stack-Edu source JSON differs"
                            ) from error
                        route, selected = _route(row)
                        counts[route] += 1
                        if selected is None:
                            continue
                        identity = canonical_sha256(
                            {
                                "source_path": parent["source_path"],
                                "source_row_index": row_index,
                                "blob_id": selected["blob_id"],
                                "content_sha256": selected["content_sha256"],
                            }
                        )
                        pending.append(
                            {
                                "schema": LOCATOR_SCHEMA,
                                "source_id": SOURCE_ID,
                                "source_repository": SOURCE_REPOSITORY,
                                "source_revision": SOURCE_REVISION,
                                "source_path": parent["source_path"],
                                "source_parent_sha256": parent["sha256"],
                                "source_row_index": row_index,
                                "source_row_identity_sha256": identity,
                                **selected,
                            }
                        )
                        identities.append(identity)
                        selected_bytes += selected["text_utf8_bytes"]
                        languages[selected["language"]] += 1
                        scores[selected["integer_score"]] += 1
                        licenses.update(selected["licenses"])
                        if len(pending) >= 256:
                            writer.write_table(
                                pa.Table.from_pylist(pending, schema=schema)
                            )
                            pending.clear()
            except (OSError, UnicodeError) as error:
                raise StackEduPracticalScanError(
                    "Stack-Edu compressed content differs"
                ) from error
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
        writer.close()
        os.replace(temporary_path, output_path)
    except BaseException:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_common_pile_stack_edu_practical_scan_shard",
        "shard_index": shard_index,
        "expected_parents": expected_parents,
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path": parent["source_path"],
            "parent_bytes": parent["bytes"],
            "parent_sha256": parent["sha256"],
        },
        "policy": {
            "accepted_integer_scores": sorted(ACCEPTED_INTEGER_SCORES),
            "accepted_languages": sorted(LANGUAGES),
            "accepted_licenses": sorted(ALLOWED_LICENSES),
            "minimum_text_bytes": MINIMUM_TEXT_BYTES,
            "maximum_text_bytes": MAXIMUM_TEXT_BYTES,
            "nonvendor_only": True,
            "nongenerated_only": True,
            "bounded_safety_policy_sha256": canonical_sha256(safety_policy()),
            "bounded_review_exclusions": sorted(_REVIEW_EXCLUSIONS),
        },
        "counts": {
            **dict(sorted(counts.items())),
            "selected_text_utf8_bytes": selected_bytes,
            "languages": dict(sorted(languages.items())),
            "integer_scores": {
                str(key): value for key, value in sorted(scores.items())
            },
            "licenses": dict(sorted(licenses.items())),
        },
        "selected": {
            "rows": len(identities),
            "text_utf8_bytes": selected_bytes,
            "ordered_identity_sha256": canonical_sha256(identities),
        },
        "output": {
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "full_parent_byte_identity_verified": True,
        "all_source_rows_accounted": counts["source_rows"] == sum(
            value for key, value in counts.items() if key != "source_rows"
        ),
        "source_text_copied": False,
        "global_exact_deduplication_complete": False,
        "official_benchmark_decontamination_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--expected-parents", type=int, default=EXPECTED_PARENTS)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = run_shard(
        args.manifest,
        args.output_root,
        args.shard_index,
        os.environ.get(args.token_env, ""),
        expected_parents=args.expected_parents,
        scratch_root=args.scratch_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

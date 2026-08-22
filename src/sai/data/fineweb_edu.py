"""Verify and convert pinned FineWeb-Edu Parquet shards into Sai raw documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sai.data.decontamination import RAW_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-fineweb-edu-conversion-receipt-v1"
BOILERPLATE_MARKERS = (
    "cookie policy",
    "privacy policy",
    "terms of use",
    "all rights reserved",
    "subscribe to our newsletter",
    "accept cookies",
)


class FineWebEduError(RuntimeError):
    """The pinned source, quality policy, Parquet schema, or output differs."""


def text_quality(text: str) -> dict[str, float | int | bool]:
    if not isinstance(text, str):
        raise FineWebEduError("FineWeb text differs")
    characters = len(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_line_fraction = (
        0.0 if not lines else (len(lines) - len(set(lines))) / len(lines)
    )
    alphabetic_fraction = (
        0.0
        if not characters
        else sum(character.isalpha() for character in text) / characters
    )
    control_fraction = (
        0.0
        if not characters
        else sum(
            unicodedata.category(character) == "Cc"
            and character not in {"\n", "\r", "\t"}
            for character in text
        )
        / characters
    )
    replacement_fraction = 0.0 if not characters else text.count("\ufffd") / characters
    lowered = text.casefold()
    boilerplate_markers = sum(marker in lowered for marker in BOILERPLATE_MARKERS)
    accepted = bool(
        400 <= characters <= 100_000
        and repeated_line_fraction <= 0.30
        and boilerplate_markers <= 1
        and alphabetic_fraction >= 0.55
        and control_fraction <= 0.001
        and replacement_fraction <= 0.0005
    )
    return {
        "characters": characters,
        "repeated_line_fraction": repeated_line_fraction,
        "boilerplate_markers": boilerplate_markers,
        "alphabetic_fraction": alphabetic_fraction,
        "control_fraction": control_fraction,
        "replacement_fraction": replacement_fraction,
        "accepted": accepted,
    }


def _manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FineWebEduError("FineWeb manifest is missing or unsafe")
    payload = json.loads(path.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "sai-pinned-hf-file-selection-v1"
        or payload.get("dataset") != "HuggingFaceFW/fineweb-edu"
        or not isinstance(payload.get("revision"), str)
        or len(payload["revision"]) != 40
        or not isinstance(payload.get("files"), list)
        or not payload["files"]
    ):
        raise FineWebEduError("FineWeb manifest differs")
    for index, row in enumerate(payload["files"]):
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size"}
            or not isinstance(row["path"], str)
            or Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or isinstance(row["size"], bool)
            or not isinstance(row["size"], int)
            or row["size"] <= 0
        ):
            raise FineWebEduError(f"FineWeb manifest row {index} differs")
    return payload


def acquire(manifest: Path, source_root: Path) -> list[Path]:
    """Download the exact manifest members and verify every byte before use."""

    payload = _manifest(manifest)
    if source_root.exists():
        raise FineWebEduError("FineWeb acquisition root already exists")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise FineWebEduError("Hugging Face Hub is required for acquisition") from error
    source_root.mkdir(parents=True)
    paths = []
    for row in payload["files"]:
        downloaded = Path(
            hf_hub_download(
                repo_id=payload["dataset"],
                filename=row["path"],
                repo_type="dataset",
                revision=payload["revision"],
                local_dir=source_root,
            )
        )
        expected = source_root / row["path"]
        if (
            downloaded.resolve() != expected.resolve()
            or not expected.is_file()
            or expected.is_symlink()
            or expected.stat().st_size != row["size"]
            or sha256_file(expected) != row["sha256"]
        ):
            raise FineWebEduError("downloaded FineWeb source shard differs")
        paths.append(expected)
    return paths


def convert(
    manifest: Path, source_root: Path, output: Path, receipt: Path
) -> dict[str, Any]:
    if output.exists() or receipt.exists():
        raise FineWebEduError("FineWeb conversion output already exists")
    payload = _manifest(manifest)
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise FineWebEduError("PyArrow is required for FineWeb conversion") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if stage.exists():
        raise FineWebEduError("FineWeb conversion staging output already exists")
    source_receipts = []
    seen_text_sha256: set[str] = set()
    scanned = accepted = score_rejected = quality_rejected = duplicates = 0
    try:
        with stage.open("w") as output_handle:
            for file_row in payload["files"]:
                path = source_root / file_row["path"]
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_size != file_row["size"]
                    or sha256_file(path) != file_row["sha256"]
                ):
                    raise FineWebEduError("FineWeb source shard content differs")
                parquet = pq.ParquetFile(path)
                required = {"text", "int_score", "url"}
                if not required.issubset(parquet.schema_arrow.names):
                    raise FineWebEduError("FineWeb Parquet columns differ")
                row_index = 0
                file_accepted = 0
                for batch in parquet.iter_batches(
                    batch_size=1_024, columns=["text", "int_score", "url"]
                ):
                    table = batch.to_pydict()
                    for text, int_score, url in zip(
                        table["text"], table["int_score"], table["url"], strict=True
                    ):
                        scanned += 1
                        current_index = row_index
                        row_index += 1
                        if (
                            isinstance(int_score, bool)
                            or not isinstance(int_score, (int, float))
                            or not math.isfinite(int_score)
                            or int_score < 4
                        ):
                            score_rejected += 1
                            continue
                        quality = text_quality(text)
                        if not quality["accepted"]:
                            quality_rejected += 1
                            continue
                        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
                        if text_sha256 in seen_text_sha256:
                            duplicates += 1
                            continue
                        seen_text_sha256.add(text_sha256)
                        host = (
                            urlsplit(url if isinstance(url, str) else "").hostname or ""
                        )
                        row = {
                            "schema": RAW_SCHEMA,
                            "text": text,
                            "source": {
                                "dataset": payload["dataset"],
                                "revision": payload["revision"],
                                "source_file": file_row["path"],
                                "row_index": current_index,
                                "license": "ODC-By-1.0_plus_Common_Crawl_terms",
                                "domain": "english",
                                "url_host": host.casefold(),
                                "int_score": int(int_score),
                                "quality_sha256": canonical_sha256(quality),
                            },
                        }
                        output_handle.write(
                            json.dumps(row, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                        accepted += 1
                        file_accepted += 1
                source_receipts.append(
                    {**file_row, "rows": row_index, "accepted": file_accepted}
                )
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    if not accepted:
        stage.unlink(missing_ok=True)
        raise FineWebEduError("FineWeb conversion admitted no documents")
    report = {
        "schema": SCHEMA,
        "status": "passed",
        "dataset": payload["dataset"],
        "revision": payload["revision"],
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "source_receipts": source_receipts,
        "quality_policy": {
            "int_score_minimum": 4,
            "characters": [400, 100_000],
            "repeated_line_fraction_maximum": 0.30,
            "boilerplate_markers_maximum": 1,
            "alphabetic_fraction_minimum": 0.55,
            "control_fraction_maximum": 0.001,
            "replacement_fraction_maximum": 0.0005,
            "boilerplate_markers": list(BOILERPLATE_MARKERS),
        },
        "scanned": scanned,
        "accepted": accepted,
        "score_rejected": score_rejected,
        "quality_rejected": quality_rejected,
        "duplicates_rejected": duplicates,
        "output": {
            "path": str(output.resolve()),
            "bytes": stage.stat().st_size,
            "sha256": sha256_file(stage),
        },
    }
    report["receipt_sha256"] = canonical_sha256(report)
    receipt_stage = receipt.with_name(f".{receipt.name}.partial.{os.getpid()}")
    receipt_stage.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(stage, output)
    os.replace(receipt_stage, receipt)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    if args.acquire:
        acquire(args.manifest, args.source_root)
    report = convert(args.manifest, args.source_root, args.output, args.receipt)
    print(
        json.dumps(
            {"status": report["status"], "receipt_sha256": report["receipt_sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

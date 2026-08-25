"""Build deterministic byte-level BPE candidates from admitted Sai documents."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sai.data.token_stream import (
    TokenStreamError,
    canonical_sha256,
    normalize_tokenizer_document,
    sha256_file,
    sha256_tree,
)

SCHEMA = "sai-tokenizer-build-manifest-v1"
SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|think|>", "<|code|>")
PRODUCTION_SIZES = {"32k": 32_000, "48k": 48_000, "64k": 64_000}


class TokenizerBuildError(RuntimeError):
    """A tokenizer source, geometry, or output differs."""


def _texts(paths: tuple[Path, ...]) -> Iterable[str]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    yield normalize_tokenizer_document(row)["text"]
                except (json.JSONDecodeError, TokenStreamError) as error:
                    raise TokenizerBuildError(
                        f"tokenizer corpus row differs: {path}"
                    ) from error


def _source_receipts(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    if not paths:
        raise TokenizerBuildError("at least one tokenizer corpus is required")
    receipts = []
    seen = set()
    for order, path in enumerate(paths):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise TokenizerBuildError(f"tokenizer corpus is missing or unsafe: {path}")
        resolved = str(path.resolve())
        if resolved in seen:
            raise TokenizerBuildError("tokenizer corpora must be unique")
        seen.add(resolved)
        receipts.append(
            {
                "order": order,
                "path": resolved,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return receipts


def build_candidates(
    corpora: list[Path], output_root: Path, *, sizes: dict[str, int]
) -> dict[str, Any]:
    """Build exact candidates into a new immutable-identity directory tree."""

    normalized_paths = tuple(corpora)
    sources = _source_receipts(normalized_paths)
    if (
        not isinstance(sizes, dict)
        or not sizes
        or any(not isinstance(name, str) or not name for name in sizes)
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 261
            for size in sizes.values()
        )
        or len(set(sizes.values())) != len(sizes)
    ):
        raise TokenizerBuildError("tokenizer candidate sizes differ")
    if output_root.exists():
        raise TokenizerBuildError("tokenizer output root already exists")
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as error:
        raise TokenizerBuildError(
            "tokenizers and transformers are required to build candidates"
        ) from error

    stage = output_root.with_name(f".{output_root.name}.partial.{os.getpid()}")
    if stage.exists():
        raise TokenizerBuildError("tokenizer staging root already exists")
    stage.mkdir(parents=True)
    candidates: dict[str, Any] = {}
    try:
        for name, size in sorted(sizes.items(), key=lambda item: item[1]):
            tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Digits(individual_digits=True),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
                ]
            )
            tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=size,
                special_tokens=list(SPECIAL_TOKENS),
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=False,
            )
            tokenizer.train_from_iterator(_texts(normalized_paths), trainer=trainer)
            if tokenizer.get_vocab_size() != size:
                raise TokenizerBuildError(f"{name} tokenizer vocabulary is incomplete")

            root = stage / name
            wrapped = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                pad_token=SPECIAL_TOKENS[0],
                bos_token=SPECIAL_TOKENS[1],
                eos_token=SPECIAL_TOKENS[2],
                clean_up_tokenization_spaces=False,
            )
            wrapped.save_pretrained(root)
            candidates[name] = {
                "name": name,
                "vocab_size": size,
                "root": name,
                "tree_sha256": sha256_tree(root),
                "files": sorted(path.name for path in root.iterdir()),
            }

        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "complete",
            "training_authorized": False,
            "candidate_build_authorized": True,
            "deterministic_source_order": True,
            "byte_fallback": True,
            "individual_digit_pretokenization": True,
            "special_tokens": list(SPECIAL_TOKENS),
            "source_receipts": sources,
            "source_identity_sha256": canonical_sha256(sources),
            "candidates": candidates,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_sizes(values: list[str] | None) -> dict[str, int]:
    """Parse explicit name=size candidates or return the production tournament."""

    if values is None:
        return dict(PRODUCTION_SIZES)
    sizes = {}
    for value in values:
        name, separator, encoded_size = value.partition("=")
        try:
            size = int(encoded_size)
        except ValueError as error:
            raise TokenizerBuildError(f"invalid tokenizer size: {value}") from error
        if (
            not separator
            or name not in PRODUCTION_SIZES
            or name in sizes
            or size != PRODUCTION_SIZES[name]
        ):
            raise TokenizerBuildError(f"invalid tokenizer size: {value}")
        sizes[name] = size
    if not sizes:
        raise TokenizerBuildError("at least one tokenizer size is required")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--size", action="append")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_candidates(
        args.corpus, args.output_root, sizes=parse_sizes(args.size)
    )
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

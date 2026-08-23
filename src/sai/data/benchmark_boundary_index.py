"""Build a non-reversible exact-shingle index of pinned public benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.decontamination import (
    _CODE,
    _WORD,
    POLICY,
    _code_shingles,
    _normalize,
    _shingles,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-official-benchmark-boundary-index-v2"
WORD_INDEX = "word_13_sha256.bin"
CODE_INDEX = "code_8_eligible_sha256.bin"
LIVEBENCH_RELEASE = "2024-11-25"
EXPECTED_ROWS = {
    "correctbench": 739,
    "humaneval_plus": 164,
    "ifeval": 541,
    "livebench": 1_000,
    "livecodebench": 1_055,
    "longbench_pro": 1_500,
    "mbpp_plus": 378,
    "mmlu_pro": 12_102,
    "musr": 756,
}
CODE_BENCHMARKS = {"humaneval_plus", "livecodebench", "mbpp_plus"}


class BenchmarkBoundaryError(RuntimeError):
    """Pinned benchmark source, projection, or exact index differs."""


@dataclass(frozen=True)
class SourceSpec:
    benchmark: str
    provider: str
    repository: str
    revision: str
    filename: str
    size: int
    content_sha256: str | None = None
    git_blob_sha1: str | None = None
    projection: str = "all_strings"


HF_SOURCES = (
    SourceSpec(
        "mmlu_pro",
        "huggingface",
        "TIGER-Lab/MMLU-Pro",
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "data/test-00000-of-00001.parquet",
        4_144_185,
        "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    ),
    SourceSpec(
        "mmlu_pro",
        "huggingface",
        "TIGER-Lab/MMLU-Pro",
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "data/validation-00000-of-00001.parquet",
        42_857,
        "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078",
    ),
    SourceSpec(
        "humaneval_plus",
        "huggingface",
        "evalplus/humanevalplus",
        "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
        "data/test-00000-of-00001-5973903632b82d40.parquet",
        2_902_210,
        "4436f5c03d77c17e0cbc57543b90665b5c1266f55a43992a5ed7922cd34a7558",
    ),
    SourceSpec(
        "mbpp_plus",
        "huggingface",
        "evalplus/mbppplus",
        "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
        "data/test-00000-of-00001-d5781c9c51e02795.parquet",
        1_129_135,
        "dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b",
    ),
    SourceSpec(
        "correctbench",
        "huggingface",
        "zeli2024/CorrectBench",
        "4927112cc54e736d169bef22d218d812c472636a",
        "CorrectBench.parquet",
        359_813,
        "4ea8e5079cb6a20278a0e36b48a6fc9c6933929b3b7ca2b5b68fddd2225aa388",
    ),
    SourceSpec(
        "livecodebench",
        "huggingface",
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "test.jsonl",
        1_252_609_773,
        "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f",
        projection="livecodebench_model_visible",
    ),
    SourceSpec(
        "livecodebench",
        "huggingface",
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "test2.jsonl",
        713_377_060,
        "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3",
        projection="livecodebench_model_visible",
    ),
    SourceSpec(
        "livecodebench",
        "huggingface",
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "test3.jsonl",
        623_360_766,
        "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79",
        projection="livecodebench_model_visible",
    ),
    SourceSpec(
        "livecodebench",
        "huggingface",
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "test4.jsonl",
        1_204_644_685,
        "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da",
        projection="livecodebench_model_visible",
    ),
    SourceSpec(
        "livecodebench",
        "huggingface",
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "test5.jsonl",
        557_699_297,
        "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214",
        projection="livecodebench_model_visible",
    ),
    SourceSpec(
        "livecodebench",
        "huggingface",
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "test6.jsonl",
        134_303_240,
        "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
        projection="livecodebench_model_visible",
    ),
    SourceSpec(
        "longbench_pro",
        "huggingface",
        "caskcsg/LongBench-Pro",
        "4996884deae51f5e5d23c88da9d857fc54e5fa15",
        "longbench_pro.json",
        531_535_940,
        "92ff05f6088e212d06c5a731ab86000b69cee6a0900cbbd524a25851e3c30de0",
    ),
    SourceSpec(
        "livebench",
        "huggingface",
        "livebench/coding",
        "a958549fdd8aa57be0a3fafe7b205ffc160ed5f4",
        "data/test-00000-of-00001.parquet",
        244_785_858,
        "5f02d01fb21672f5d84169f940adab46ff3ca09b9159fd42fdd289bc9be23502",
        projection="livebench_release",
    ),
    SourceSpec(
        "livebench",
        "huggingface",
        "livebench/data_analysis",
        "31b9661ff678df9958e2f7fa228427f4c858c1a1",
        "data/test-00000-of-00001.parquet",
        144_796,
        "fb86a7a02fa9eabf785d9e8af85955990cf6d228cc9d3f805a54f863bc8c4c52",
        projection="livebench_release",
    ),
    SourceSpec(
        "livebench",
        "huggingface",
        "livebench/instruction_following",
        "0868379c4b5cf62aeacaf8be4f08fced815c81bb",
        "data/test-00000-of-00001.parquet",
        537_024,
        "a9bb97bbaf8788142c310bcb33d50e2f6f5df8cbd8b8c3db677816b06f0f4f25",
        projection="livebench_release",
    ),
    SourceSpec(
        "livebench",
        "huggingface",
        "livebench/math",
        "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb",
        "data/test-00000-of-00001.parquet",
        185_605,
        "3d365cad1f9b8d7c5416d63866653d6854b270d9c93b51323d405ed5fd51df54",
        projection="livebench_release",
    ),
    SourceSpec(
        "livebench",
        "huggingface",
        "livebench/reasoning",
        "6fc6498a5dfba553f69f4413feabade1f1a2d384",
        "data/test-00000-of-00001.parquet",
        88_219,
        "4204bb94c812690ef8ba5f4c1f10b5b1082ca0b7bc532166834f798aa56e2a3c",
        projection="livebench_release",
    ),
    SourceSpec(
        "livebench",
        "huggingface",
        "livebench/language",
        "3ada32a2e53d5e04e57fa503384cb85ce9116c40",
        "data/test-00000-of-00001.parquet",
        287_876,
        "76ba142afd242ca02d6baa8bb737608d2b416674f311d8f9d798b4e3908a499c",
        projection="livebench_release",
    ),
)

GITHUB_SOURCES = (
    SourceSpec(
        "ifeval",
        "github",
        "google-research/google-research",
        "589e977488f21a336a3d3da9b96da91ddbcf935e",
        "instruction_following_eval/data/input_data.jsonl",
        207_111,
        git_blob_sha1="cbe52f6eecf3986fdac745b4acba4da1408eb146",
    ),
    SourceSpec(
        "musr",
        "github",
        "Zayne-sprague/MuSR",
        "b1f4d4168a9cfc6760e8b74d728e4516023dfaa5",
        "datasets/murder_mystery.json",
        44_386_771,
        git_blob_sha1="990ba68bdbc934e0d6608723ae66237ea8f0ac41",
        projection="musr",
    ),
    SourceSpec(
        "musr",
        "github",
        "Zayne-sprague/MuSR",
        "b1f4d4168a9cfc6760e8b74d728e4516023dfaa5",
        "datasets/object_placements.json",
        5_079_054,
        git_blob_sha1="bc083672c66a0fca5fa7de6b43a062e9a38ec3f5",
        projection="musr",
    ),
    SourceSpec(
        "musr",
        "github",
        "Zayne-sprague/MuSR",
        "b1f4d4168a9cfc6760e8b74d728e4516023dfaa5",
        "datasets/team_allocation.json",
        5_620_528,
        git_blob_sha1="ccd8321491423b0578e56ba25ed4067eda00c9e0",
        projection="musr",
    ),
)

SOURCE_SPECS = HF_SOURCES + GITHUB_SOURCES


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()  # noqa: S324 - required Git object identity replay
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_url(spec: SourceSpec) -> str:
    quoted = urllib.parse.quote(spec.filename, safe="/")
    if spec.provider == "huggingface":
        return (
            f"https://huggingface.co/datasets/{spec.repository}/resolve/"
            f"{spec.revision}/{quoted}?download=true"
        )
    if spec.provider == "github":
        return (
            f"https://raw.githubusercontent.com/{spec.repository}/"
            f"{spec.revision}/{quoted}"
        )
    raise BenchmarkBoundaryError("benchmark provider differs")


def acquire_source(spec: SourceSpec, target: Path) -> dict[str, Any]:
    """Download one public source without retaining a cache duplicate."""

    if target.exists() or target.is_symlink() or spec.size <= 0:
        raise BenchmarkBoundaryError("benchmark source target differs")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{target.name}.partial.{os.getpid()}")
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(
        _source_url(spec), headers={"User-Agent": "sai-boundary-index/1"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=300) as response,
            stage.open("xb") as handle,
        ):
            for chunk in iter(lambda: response.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if size != spec.size or (
            spec.content_sha256 is not None
            and digest.hexdigest() != spec.content_sha256
        ):
            raise BenchmarkBoundaryError("downloaded benchmark source differs")
        os.replace(stage, target)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    if spec.git_blob_sha1 is not None and _git_blob_sha1(target) != spec.git_blob_sha1:
        target.unlink(missing_ok=True)
        raise BenchmarkBoundaryError("downloaded Git benchmark object differs")
    return {
        "benchmark": spec.benchmark,
        "provider": spec.provider,
        "repository": spec.repository,
        "revision": spec.revision,
        "filename": spec.filename,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "git_blob_sha1": spec.git_blob_sha1,
        "projection": spec.projection,
    }


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if len(value.strip()) >= 8 else []
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, dict):
        return [text for key in sorted(value) for text in _strings(value[key])]
    return []


def _json_array(path: Path) -> Iterator[Any]:
    """Stream a top-level JSON array without loading a long-context board."""

    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = ended = False
        eof = False
        while not ended:
            if position > (1 << 20):
                buffer = buffer[position:]
                position = 0
            while position >= len(buffer) and not eof:
                chunk = handle.read(1 << 20)
                eof = not chunk
                buffer += chunk
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer) or buffer[position] != "[":
                    raise BenchmarkBoundaryError("benchmark JSON array differs")
                position += 1
                started = True
                continue
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                ended = True
                continue
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    yield value
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise BenchmarkBoundaryError(
                            "benchmark JSON array is truncated"
                        ) from error
                    buffer = buffer[position:]
                    position = 0
                    chunk = handle.read(1 << 20)
                    eof = not chunk
                    buffer += chunk
        if buffer[position:].strip():
            raise BenchmarkBoundaryError("benchmark JSON array has trailing data")


def _records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise BenchmarkBoundaryError("PyArrow is required") from error
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=128):
            yield from batch.to_pylist()
        return
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise BenchmarkBoundaryError("benchmark JSONL row differs")
                    yield value
        return
    if path.suffix == ".json":
        for value in _json_array(path):
            if not isinstance(value, dict):
                raise BenchmarkBoundaryError("benchmark JSON row differs")
            yield value
        return
    raise BenchmarkBoundaryError("benchmark source format differs")


def projected_strings(spec: SourceSpec, row: dict[str, Any]) -> list[list[str]]:
    """Return one or more benchmark-item string groups from an upstream row."""

    if spec.projection == "livecodebench_model_visible":
        return [
            _strings(
                {
                    key: row.get(key)
                    for key in (
                        "question_title",
                        "question_content",
                        "starter_code",
                        "public_test_cases",
                    )
                }
            )
        ]
    if spec.projection == "livebench_release":
        released = str(row.get("livebench_release_date", ""))[:10]
        removed = str(row.get("livebench_removal_date", ""))[:10]
        if released > LIVEBENCH_RELEASE or (removed and removed <= LIVEBENCH_RELEASE):
            return []
        return [_strings(row)]
    if spec.projection == "musr":
        context = row.get("context")
        questions = row.get("questions")
        if not isinstance(context, str) or not isinstance(questions, list):
            raise BenchmarkBoundaryError("MuSR source row differs")
        groups = []
        for question in questions:
            if not isinstance(question, dict):
                raise BenchmarkBoundaryError("MuSR question differs")
            groups.append(_strings({"context": context, "question": question}))
        return groups
    return [_strings(row)]


def _iter_digests(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while chunk := handle.read(32):
            if len(chunk) != 32:
                raise BenchmarkBoundaryError("digest chunk is truncated")
            yield chunk


class DigestChunks:
    """External-sort exact 256-bit digests with bounded memory."""

    def __init__(self, root: Path, label: str, maximum_buffered: int) -> None:
        self.root = root
        self.label = label
        self.maximum_buffered = maximum_buffered
        self.buffer: set[bytes] = set()
        self.chunks: list[Path] = []
        self.observations = 0

    def add(self, values: Iterable[bytes]) -> None:
        for value in values:
            if not isinstance(value, bytes) or len(value) != 32:
                raise BenchmarkBoundaryError("boundary digest differs")
            self.buffer.add(value)
            self.observations += 1
            if len(self.buffer) >= self.maximum_buffered:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.root / f"{self.label}.{len(self.chunks):06d}.bin"
        with path.open("xb") as handle:
            for digest in sorted(self.buffer):
                handle.write(digest)
            handle.flush()
            os.fsync(handle.fileno())
        self.chunks.append(path)
        self.buffer.clear()

    def finalize(self, output: Path) -> int:
        self.flush()
        if output.exists() or not self.chunks:
            raise BenchmarkBoundaryError("boundary digest output differs")
        unique = 0
        previous = None
        with output.open("xb") as handle:
            streams = [_iter_digests(path) for path in self.chunks]
            for digest in heapq.merge(*streams):
                if digest != previous:
                    handle.write(digest)
                    unique += 1
                    previous = digest
            handle.flush()
            os.fsync(handle.fileno())
        return unique


def build_boundary_index(
    output_root: Path,
    scratch_root: Path,
    *,
    source_specs: tuple[SourceSpec, ...] = SOURCE_SPECS,
    expected_rows: dict[str, int] = EXPECTED_ROWS,
    maximum_buffered_digests: int = 250_000,
) -> dict[str, Any]:
    """Acquire, project, hash, seal, and delete every raw benchmark source."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or scratch_root.exists()
        or scratch_root.is_symlink()
        or not source_specs
        or isinstance(maximum_buffered_digests, bool)
        or not 1 <= maximum_buffered_digests <= 10_000_000
    ):
        raise BenchmarkBoundaryError("benchmark boundary geometry differs")
    stage = output_root.with_name(f".{output_root.name}.partial.{os.getpid()}")
    if stage.exists():
        raise BenchmarkBoundaryError("benchmark boundary stage exists")
    stage.mkdir(parents=True)
    scratch_root.mkdir(parents=True)
    chunks_root = scratch_root / "chunks"
    chunks_root.mkdir()
    word = DigestChunks(chunks_root, "word", maximum_buffered_digests)
    code = DigestChunks(chunks_root, "code", maximum_buffered_digests)
    rows = {key: 0 for key in expected_rows}
    source_receipts = []
    strings = 0
    try:
        for order, spec in enumerate(source_specs):
            if spec.benchmark not in expected_rows:
                raise BenchmarkBoundaryError("benchmark source is outside coverage")
            source_path = (
                scratch_root / f"source-{order:05d}{Path(spec.filename).suffix}"
            )
            source_receipt = acquire_source(spec, source_path)
            source_rows = boundary_rows = source_strings = 0
            try:
                for source_row in _records(source_path):
                    source_rows += 1
                    for group in projected_strings(spec, source_row):
                        unique_group = list(dict.fromkeys(group))
                        if not unique_group:
                            raise BenchmarkBoundaryError(
                                "benchmark row produced no boundary strings"
                            )
                        boundary_rows += 1
                        rows[spec.benchmark] += 1
                        strings += len(unique_group)
                        source_strings += len(unique_group)
                        for text in unique_group:
                            normalized = _normalize(text)
                            word.add(
                                _shingles(
                                    _WORD.findall(normalized),
                                    POLICY["word_shingle_tokens"],
                                )
                            )
                            if spec.benchmark in CODE_BENCHMARKS:
                                code.add(_code_shingles(_CODE.findall(normalized)))
            finally:
                source_path.unlink(missing_ok=True)
            source_receipts.append(
                {
                    "order": order,
                    **source_receipt,
                    "source_rows": source_rows,
                    "boundary_rows": boundary_rows,
                    "boundary_strings": source_strings,
                    "raw_source_persisted": False,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "benchmark_boundary_source_complete",
                        "order": order,
                        "benchmark": spec.benchmark,
                        "source_rows": source_rows,
                        "boundary_rows": boundary_rows,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if rows != expected_rows:
            raise BenchmarkBoundaryError(
                f"benchmark boundary cardinality differs: {rows}"
            )
        word_path = stage / WORD_INDEX
        code_path = stage / CODE_INDEX
        unique_word = word.finalize(word_path)
        unique_code = code.finalize(code_path)
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "policy": POLICY,
            "policy_sha256": canonical_sha256(POLICY),
            "benchmark_rows": rows,
            "boundary_rows": sum(rows.values()),
            "boundary_strings": strings,
            "source_files": source_receipts,
            "source_files_sha256": canonical_sha256(source_receipts),
            "word_index": {
                "file": WORD_INDEX,
                "digest_bytes": 32,
                "observed_shingles": word.observations,
                "unique_shingles": unique_word,
                "bytes": word_path.stat().st_size,
                "sha256": sha256_file(word_path),
            },
            "code_index": {
                "file": CODE_INDEX,
                "digest_bytes": 32,
                "observed_shingles": code.observations,
                "unique_shingles": unique_code,
                "bytes": code_path.stat().st_size,
                "sha256": sha256_file(code_path),
            },
            "index_order": "strictly_increasing_unsigned_32_byte_sha256",
            "livecodebench_projection": (
                "question_title_question_content_starter_code_public_tests; "
                "private_grading_payload_not_redistributed"
            ),
            "known_gap": (
                "RULER is generator-defined and requires a separately pinned "
                "tokenizer-bound generation receipt before joining this index"
            ),
            "raw_benchmark_text_persisted": False,
            "benchmark_contamination_gate_ready": True,
            "training_data": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--maximum-buffered-digests", type=int, default=250_000)
    args = parser.parse_args()
    payload = build_boundary_index(
        args.output_root,
        args.scratch_root,
        maximum_buffered_digests=args.maximum_buffered_digests,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

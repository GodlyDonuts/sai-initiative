from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.freeze import DATA_ROLES, ROW_SCHEMA, DataFreezeError, freeze


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def training_row(role: str, index: int, prompt: str, response: str | None) -> dict:
    modes = {
        "skill_direct": "direct",
        "skill_deliberate": "deliberate",
        "behavior_replay": "replay",
        "rl_prompts": "rl_prompt",
    }
    return {
        "schema": ROW_SCHEMA,
        "role": role,
        "mode": modes[role],
        "prompt": prompt,
        "response": response,
        "source": {
            "dataset": f"source-{role}",
            "row_id": str(index),
            "license": "Apache-2.0",
        },
        "verification": {
            "passed": True,
            "kind": "rule_verifier" if role == "rl_prompts" else "exact_answer",
            "evidence_sha256": hashlib.sha256(f"{role}-{index}".encode()).hexdigest(),
        },
    }


def fixture(tmp_path: Path):
    benchmarks = {}
    for name in ("humaneval_plus", "mbpp_plus", "ifeval", "musr", "correctbench"):
        path = tmp_path / f"benchmark-{name}.jsonl"
        write_rows(path, [{"id": f"{name}-1", "prompt": f"sealed {name} problem"}])
        benchmarks[name] = path
    sources = {}
    for index, role in enumerate(sorted(DATA_ROLES)):
        path = tmp_path / f"{role}.jsonl"
        response = None if role == "rl_prompts" else f"verified answer {index}"
        write_rows(path, [training_row(role, index, f"unique {role} task", response)])
        sources[role] = path
    return sources, benchmarks


def test_freeze_writes_all_roles_and_no_training_authorization(tmp_path: Path) -> None:
    sources, benchmarks = fixture(tmp_path)
    output = tmp_path / "frozen"
    report = freeze(
        sources,
        benchmarks,
        output,
        expected_benchmark_rows={name: 1 for name in benchmarks},
    )
    assert set(report["roles"]) == DATA_ROLES
    assert all(receipt["rows"] == 1 for receipt in report["roles"].values())
    assert not report["training_authorized"]
    assert (output / "freeze_report.json").is_file()
    for role in DATA_ROLES:
        row = json.loads((output / f"{role}.jsonl").read_text())
        assert row["role"] == role
        assert len(row["identity_sha256"]) == 64


def test_filter_covers_prompt_and_response(tmp_path: Path) -> None:
    sources, benchmarks = fixture(tmp_path)
    direct = sources["skill_direct"]
    write_rows(
        direct,
        [
            training_row("skill_direct", 1, "sealed musr problem", "answer"),
            training_row(
                "skill_direct", 2, "unrelated clean task", "sealed ifeval problem"
            ),
            training_row("skill_direct", 3, "surviving task", "verified result"),
        ],
    )
    report = freeze(
        sources,
        benchmarks,
        tmp_path / "frozen",
        expected_benchmark_rows={name: 1 for name in benchmarks},
    )
    receipt = report["roles"]["skill_direct"]
    assert receipt["rows"] == 1
    assert receipt["benchmark_overlap_drops"] == {
        "exact:prompt": 1,
        "exact:response": 1,
    }


def test_global_duplicate_prompt_is_removed(tmp_path: Path) -> None:
    sources, benchmarks = fixture(tmp_path)
    duplicate = "same model visible prompt"
    write_rows(
        sources["behavior_replay"],
        [training_row("behavior_replay", 1, duplicate, "parent response")],
    )
    write_rows(
        sources["skill_direct"],
        [training_row("skill_direct", 2, duplicate, "verified response")],
    )
    with pytest.raises(DataFreezeError, match="has no accepted rows"):
        freeze(
            sources,
            benchmarks,
            tmp_path / "frozen",
            expected_benchmark_rows={name: 1 for name in benchmarks},
        )


def test_unverified_row_cannot_enter_population(tmp_path: Path) -> None:
    sources, benchmarks = fixture(tmp_path)
    row = training_row("skill_deliberate", 1, "reason carefully", "trace")
    row["verification"]["passed"] = False
    write_rows(sources["skill_deliberate"], [row])
    with pytest.raises(DataFreezeError, match="has no accepted rows"):
        freeze(
            sources,
            benchmarks,
            tmp_path / "frozen",
            expected_benchmark_rows={name: 1 for name in benchmarks},
        )


def test_incomplete_benchmark_board_fails_closed(tmp_path: Path) -> None:
    sources, benchmarks = fixture(tmp_path)
    with pytest.raises(DataFreezeError, match="row geometry"):
        freeze(sources, benchmarks, tmp_path / "frozen")

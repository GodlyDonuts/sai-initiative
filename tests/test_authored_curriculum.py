from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from sai.data.authored_curriculum import (
    AuthoredCurriculumError,
    build,
    validate,
)

RUST_REVISION = "1" * 40
PYTHON_REVISION = "2" * 40


def _archive(
    path: Path, members: dict[str, bytes], *, symlink: tuple[str, str] | None = None
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, encoded in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(encoded)
            info.mode = 0o444
            archive.addfile(info, io.BytesIO(encoded))
        if symlink is not None:
            info = tarfile.TarInfo(symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            archive.addfile(info)


def _sources(tmp_path: Path, *, python_symlink: bool = False) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rust_root = f"book-{RUST_REVISION}"
    rust_summary = ["# Book", "", "[Title](title-page.md)"]
    rust_members = {
        f"{rust_root}/LICENSE-APACHE": b"apache\n",
        f"{rust_root}/LICENSE-MIT": b"mit\n",
        f"{rust_root}/src/title-page.md": b"# Title\n",
    }
    for index in range(1, 80):
        name = f"ch{index:02d}-00.md"
        rust_summary.append(f"- [Chapter {index}]({name})")
        rust_members[f"{rust_root}/src/{name}"] = (
            f"# Chapter {index}\n\n```rust\nfn main() {{}}\n```\n".encode()
        )
    rust_members[f"{rust_root}/src/SUMMARY.md"] = (
        "\n".join(rust_summary) + "\n"
    ).encode()
    rust = tmp_path / "rust.tar.gz"
    _archive(rust, rust_members)

    python_root = f"cpython-{PYTHON_REVISION}"
    tutorial_names = [
        "appetite",
        "interpreter",
        "introduction",
        "controlflow",
        "datastructures",
        "modules",
        "inputoutput",
        "errors",
        "classes",
        "stdlib",
        "stdlib2",
        "venv",
        "whatnow",
        "interactive",
        "floatingpoint",
        "appendix",
    ]
    python_index = "Tutorial\n========\n\n.. toctree::\n   :numbered:\n\n" + "".join(
        f"   {name}.rst\n" for name in tutorial_names
    )
    python_members = {
        f"{python_root}/LICENSE": b"python license\n",
        f"{python_root}/Doc/tutorial/index.rst": python_index.encode(),
    }
    for name in tutorial_names:
        chapter = (
            f"{name.title()}\n{'=' * len(name)}\n\n"
            ".. code-block:: python\n\n   print('ok')\n"
        )
        python_members[f"{python_root}/Doc/tutorial/{name}.rst"] = (chapter).encode()
    python = tmp_path / "python.tar.gz"
    symlink = (
        (
            f"{python_root}/Doc/tutorial/appetite.rst",
            "../../LICENSE",
        )
        if python_symlink
        else None
    )
    if python_symlink:
        del python_members[f"{python_root}/Doc/tutorial/appetite.rst"]
    _archive(python, python_members, symlink=symlink)
    return rust, python


def test_build_preserves_progression_and_holds_training(tmp_path: Path) -> None:
    rust, python = _sources(tmp_path)
    output = tmp_path / "candidate.jsonl"
    receipt = tmp_path / "receipt.json"
    report = build(
        rust_archive=rust,
        rust_revision=RUST_REVISION,
        python_archive=python,
        python_revision=PYTHON_REVISION,
        output=output,
        receipt_output=receipt,
    )
    assert report["training_authorized"] is False
    assert report["four_b_training_authorized"] is False
    assert report["summary"]["rows"] == 96
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["source_path"] == "src/title-page.md"
    assert rows[79]["source_path"] == "src/ch79-00.md"
    assert rows[80]["source_path"] == "Doc/tutorial/appetite.rst"
    assert rows[80]["required_prior_concepts"] == ["programming_foundations"]
    assert ".. code-block:: python" in rows[80]["text"]
    assert output.stat().st_mode & 0o222 == 0
    assert validate(output, receipt)["receipt_sha256"] == report["receipt_sha256"]


def test_rejects_selected_symlink_and_create_overwrite(tmp_path: Path) -> None:
    rust, python = _sources(tmp_path, python_symlink=True)
    with pytest.raises(AuthoredCurriculumError, match="missing or unsafe"):
        build(
            rust_archive=rust,
            rust_revision=RUST_REVISION,
            python_archive=python,
            python_revision=PYTHON_REVISION,
            output=tmp_path / "candidate.jsonl",
            receipt_output=tmp_path / "receipt.json",
        )

    rust, python = _sources(tmp_path / "second")
    output = tmp_path / "exists.jsonl"
    output.write_text("occupied")
    with pytest.raises(AuthoredCurriculumError, match="boundary"):
        build(
            rust_archive=rust,
            rust_revision=RUST_REVISION,
            python_archive=python,
            python_revision=PYTHON_REVISION,
            output=output,
            receipt_output=tmp_path / "new-receipt.json",
        )


def test_validation_rejects_tamper(tmp_path: Path) -> None:
    rust, python = _sources(tmp_path)
    output = tmp_path / "candidate.jsonl"
    receipt = tmp_path / "receipt.json"
    build(
        rust_archive=rust,
        rust_revision=RUST_REVISION,
        python_archive=python,
        python_revision=PYTHON_REVISION,
        output=output,
        receipt_output=receipt,
    )
    output.chmod(0o644)
    output.write_text(
        output.read_text().replace("programming_foundations", "advanced_python", 1)
    )
    with pytest.raises(AuthoredCurriculumError, match="receipt differs"):
        validate(output, receipt)

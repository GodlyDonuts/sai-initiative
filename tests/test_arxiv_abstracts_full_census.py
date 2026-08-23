import gzip
import hashlib
import json

from sai.data.arxiv_abstracts_audit_population import REPOSITORY, REVISION
from sai.data.arxiv_abstracts_full_census import _scan_parent


def _row(path: str, line: int, native_id: str, text: str) -> dict:
    return {
        "created": "2024-01-01T00:00:00",
        "id": native_id,
        "metadata": {
            "license": "Creative Commons Zero - Public Domain - test",
            "provenance": f"{path}:{line}",
        },
        "source": "arxiv-abstracts",
        "text": text,
    }


def test_parent_scan_counts_audit_short_and_exact_duplicate(tmp_path) -> None:
    name = "test-parent.json.gz"
    eligible = "A rigorous scientific abstract with evidence. " * 12
    audit_text = "A held-out validation abstract with evidence. " * 12
    rows = [
        _row(name, 1, "one", eligible),
        _row(name, 2, "two", "short"),
        _row(name, 3, "three", eligible),
        _row(name, 4, "four", audit_text),
        _row(name, 5, "five", "A license edge row with enough content. " * 12),
    ]
    rows[-1]["metadata"]["license"] = "upstream terms unknown"
    rows[3]["metadata"]["license"] = "held-out row must be excluded first"
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    parent = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    ordered = hashlib.sha256()
    receipt = _scan_parent(
        path,
        parent,
        {
            (REPOSITORY, name, "physical_line", 4): hashlib.sha256(
                audit_text.strip().encode()
            ).hexdigest()
        },
        frozenset(),
        set(),
        set(),
        set(),
        set(),
        ordered,
    )
    assert receipt["scanned_rows"] == 5
    assert receipt["text_rows"] == 5
    assert receipt["non_cc0_declaration_rows"] == 1
    assert receipt["validated_source_rows"] == 3
    assert receipt["audit_excluded_rows"] == 1
    assert receipt["audit_position_excluded_rows"] == 1
    assert receipt["short_rows"] == 1
    assert receipt["exact_duplicate_rows"] == 1
    assert receipt["mechanically_eligible_unique_rows"] == 1
    assert receipt["source_text_persisted"] is False
    assert ordered.hexdigest() != hashlib.sha256().hexdigest()


def test_parent_scan_excludes_audited_content_at_another_position(tmp_path) -> None:
    name = "test-parent.json.gz"
    held_out_text = "A held-out validation abstract with evidence. " * 12
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(name, 1, "one", held_out_text)) + "\n")
    parent = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    receipt = _scan_parent(
        path,
        parent,
        {},
        frozenset({hashlib.sha256(held_out_text.strip().encode()).hexdigest()}),
        set(),
        set(),
        set(),
        set(),
        hashlib.sha256(),
    )
    assert receipt["audit_excluded_rows"] == 1
    assert receipt["audit_content_excluded_rows"] == 1
    assert receipt.get("mechanically_eligible_unique_rows", 0) == 0


def test_parent_scan_keeps_physical_and_source_provenance_coordinates_distinct(
    tmp_path,
) -> None:
    name = "test-parent.json.gz"
    first = "A first rigorous scientific abstract with evidence. " * 12
    held_out = "A held-out rigorous scientific abstract with evidence. " * 12
    final = "A final rigorous scientific abstract with evidence. " * 12
    rows = [
        _row(name, 1, "one", first),
        _row(name, 3, "two", held_out),
        _row(name, 4, "three", final),
    ]
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    parent = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    observed_positions = set()
    receipt = _scan_parent(
        path,
        parent,
        {
            (REPOSITORY, name, "source_provenance", 3): hashlib.sha256(
                held_out.strip().encode()
            ).hexdigest()
        },
        frozenset(),
        observed_positions,
        set(),
        set(),
        set(),
        hashlib.sha256(),
    )
    assert receipt["source_provenance_gap_positions"] == 1
    assert receipt["provenance_physical_line_delta_rows"] == 2
    assert receipt["maximum_source_provenance_line"] == 4
    assert receipt["audit_position_excluded_rows"] == 1
    assert receipt["audit_position_excluded_identities"] == 1
    assert observed_positions == {(REPOSITORY, name, "source_provenance", 3)}

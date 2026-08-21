from __future__ import annotations

import json
from pathlib import Path

from sai.tokenizer.capacity import audit, unsupported_script


class Tokenizer:
    pieces = {
        0: "<eos>",
        1: "hello",
        2: " world",
        3: "变量",
        4: "функция",
        5: "+",
        6: "π",
    }
    all_special_ids = [0]

    def get_vocab(self):
        return {piece: token_id for token_id, piece in self.pieces.items()}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "hello world": [1, 2],
            "hello+π": [1, 5, 6],
        }[text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[token_id]


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_script_classifier_preserves_mixed_and_technical_tokens() -> None:
    assert unsupported_script("变量") == "CJK"
    assert unsupported_script("функция") == "CYRILLIC"
    assert unsupported_script("hello") is None
    assert unsupported_script("x变量") is None
    assert unsupported_script("π") is None


def test_audit_protects_used_tokens_and_quantifies_unused_scripts(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    write(corpus, [{"question": "hello world", "response": "hello+π"}])
    write(evaluation, [{"prompt": "hello+π"}])
    payload = audit(
        Tokenizer(),
        [corpus],
        [evaluation],
        hidden_size=8,
        tied_embeddings=False,
    )
    candidate = payload["candidate"]
    assert candidate["removable_token_count"] == 2
    assert candidate["removable_by_script"] == {"CJK": 1, "CYRILLIC": 1}
    assert candidate["estimated_parameters_recovered"] == 32
    assert {row["id"] for row in candidate["removable_tokens"]} == {3, 4}
    assert payload["candidate_build_authorized"]
    assert not payload["scientific_training_authorized"]

"""Build an offline blinded workspace for FineMath quality review."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.finemath_filter_ladder import REVIEW_SCHEMA, validate_ladder
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-finemath-human-review-workspace-v1"
REVIEW_ROW_SCHEMA = "sai-finemath-human-quality-review-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FineMathReviewWorkspaceError(RuntimeError):
    """The FineMath ladder, offline workspace, or receipt differs."""


def _embedded_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _read_packet(
    path: Path, *, expected_rows: int
) -> tuple[list[dict[str, Any]], bytes]:
    encoded = _read_regular_bytes(path, maximum_bytes=32 << 20)
    rows: list[dict[str, Any]] = []
    try:
        for line in encoded.decode().splitlines():
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "schema",
                    "review_identity_sha256",
                    "source_url",
                    "text_sha256",
                    "text",
                }
                or row["schema"] != REVIEW_SCHEMA
                or not _HEX64.fullmatch(row["review_identity_sha256"])
                or not _HEX64.fullmatch(row["text_sha256"])
                or not isinstance(row["text"], str)
                or not row["text"]
                or hashlib.sha256(row["text"].encode()).hexdigest()
                != row["text_sha256"]
            ):
                raise ValueError
            rows.append(row)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise FineMathReviewWorkspaceError("FineMath blind packet differs") from error
    identities = [row["review_identity_sha256"] for row in rows]
    if len(rows) != expected_rows or len(set(identities)) != len(identities):
        raise FineMathReviewWorkspaceError("FineMath blind packet differs")
    return rows, encoded


def _html_document(
    *, rows: list[dict[str, Any]], workspace_identity_sha256: str
) -> bytes:
    data = {
        "workspace_identity_sha256": workspace_identity_sha256,
        "packet": [
            {
                "review_identity_sha256": row["review_identity_sha256"],
                "text": row["text"],
            }
            for row in rows
        ],
        "review_row_schema": REVIEW_ROW_SCHEMA,
        "minimum_evidence_codepoints": 12,
        "acceptance_clarity_floor_ppm": 800_000,
        "decisions": ["accept", "reject", "uncertain"],
        "correctness": ["correct", "incorrect", "uncertain"],
        "structures": ["explanatory", "answer_only", "incoherent", "uncertain"],
        "defects": [
            "answer_only",
            "duplicated_boilerplate",
            "incoherent_prose",
            "incorrect_math",
            "incomplete_solution",
            "low_value_repetition",
            "non_english_or_garbled",
            "relies_on_missing_context",
            "source_noise",
            "unsafe_or_advertising",
        ],
    }
    encoded = _embedded_json(data)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:;">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sai blinded FineMath review</title>
<style>
body{{margin:0;font:15px system-ui;color:#17202a;background:#f4f6f7}}header{{padding:12px 18px;background:#17202a;color:white}}main{{display:grid;grid-template-columns:220px 1fr 1fr;height:calc(100vh - 72px)}}aside,section{{overflow:auto;padding:12px;border-right:1px solid #ccd1d1}}button{{margin:2px;padding:6px 9px}}button.done{{background:#d5f5e3}}pre{{white-space:pre-wrap;word-break:break-word;background:white;padding:12px}}textarea{{width:100%;height:48vh;font:13px ui-monospace,monospace}}input{{padding:5px}}.error{{color:#922b21;font-weight:700}}.ok{{color:#196f3d;font-weight:700}}
</style></head><body>
<header><strong>Sai blind FineMath quality review</strong> — language score, score stratum, source URL, and threshold candidates are hidden.</header>
<main><aside><label>Reviewer pseudonym <input id="reviewer" autocomplete="off" pattern="[A-Za-z0-9_-]{{3,64}}"></label><div id="progress"></div><div id="rows"></div></aside>
<section><h2 id="heading"></h2><pre id="source"></pre></section>
<section><p>Judge correctness, clarity, explanatory value, and self-containment from the text only. Evidence must be a unique literal quote. An accepted row must be correct, explanatory, self-contained, clarity ≥800000, and defect-free.</p><textarea id="draft"></textarea><div><button id="save">Save row</button><button id="mark">Mark reviewed</button><button id="backup">Export progress</button><button id="restore">Import progress</button><input id="progress-file" type="file" accept="application/json,.json" hidden><button id="export">Export complete JSONL</button></div><p id="message"></p></section></main>
<script id="sai-data" type="application/json">{encoded}</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('sai-data').textContent);
const blank=()=>({{quality_decision:null,mathematical_correctness:null,instructional_structure:null,self_contained:null,english_clarity_ppm:null,defects:[],evidence_quotes:[]}});
const states=data.packet.map(()=>({{draft:blank(),reviewed:false}}));let current=0;
const $=id=>document.getElementById(id);const count=(text,part)=>text.split(part).length-1;
function reviewer(){{const value=$('reviewer').value.trim();if(!/^[A-Za-z0-9_-]{{3,64}}$/.test(value))throw Error('Reviewer pseudonym must be 3-64 letters, digits, underscores, or hyphens');return value}}
function normalize(value,text){{
 const keys=['defects','english_clarity_ppm','evidence_quotes','instructional_structure','mathematical_correctness','quality_decision','self_contained'];
 if(!value||typeof value!=='object'||Array.isArray(value)||Object.keys(value).sort().join('|')!==keys.join('|'))throw Error('Draft fields differ');
 if(!data.decisions.includes(value.quality_decision)||!data.correctness.includes(value.mathematical_correctness)||!data.structures.includes(value.instructional_structure))throw Error('Choose valid decision fields');
 if(typeof value.self_contained!=='boolean')throw Error('self_contained must be true or false');
 if(!Number.isInteger(value.english_clarity_ppm)||value.english_clarity_ppm<0||value.english_clarity_ppm>1000000)throw Error('english_clarity_ppm must be an integer from 0 to 1000000');
 for(const field of ['defects','evidence_quotes'])if(!Array.isArray(value[field]))throw Error(field+' must be an array');
 value.defects=[...new Set(value.defects)].sort();if(value.defects.some(x=>!data.defects.includes(x)))throw Error('Unknown defect');
 value.evidence_quotes=[...new Set(value.evidence_quotes)];if(!value.evidence_quotes.length)throw Error('At least one evidence quote is required');
 for(const quote of value.evidence_quotes)if(typeof quote!=='string'||[...quote].length<data.minimum_evidence_codepoints||count(text,quote)!==1)throw Error('Every evidence quote must be a unique literal source span');
 if(value.quality_decision==='accept'&&(value.mathematical_correctness!=='correct'||value.instructional_structure!=='explanatory'||value.self_contained!==true||value.english_clarity_ppm<data.acceptance_clarity_floor_ppm||value.defects.length))throw Error('Accept does not satisfy the frozen quality rule');
 if(value.quality_decision==='reject'&&!value.defects.length)throw Error('Reject requires at least one defect');
 return value;
}}
function store(mark){{const row=data.packet[current];const value=normalize(JSON.parse($('draft').value),row.text);states[current].draft=value;if(mark)states[current].reviewed=true;render();message(mark?'Row marked reviewed':'Row saved',true)}}
function message(text,ok){{$('message').textContent=text;$('message').className=ok?'ok':'error'}}
function download(name,text,type){{const blob=new Blob([text],{{type}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href)}}
function backup(){{try{{const rid=reviewer();states[current].draft=JSON.parse($('draft').value);const payload={{schema:'sai-finemath-human-review-progress-v1',workspace_identity_sha256:data.workspace_identity_sha256,reviewer_id:rid,rows:states.map((state,index)=>({{review_identity_sha256:data.packet[index].review_identity_sha256,reviewed:state.reviewed,draft:state.draft}}))}};download('sai-finemath-human-review-progress.json',JSON.stringify(payload,null,2)+'\\n','application/json');message('Offline progress exported',true)}}catch(e){{message(e.message,false)}}}}
async function restore(file){{try{{const payload=JSON.parse(await file.text());if(!payload||payload.schema!=='sai-finemath-human-review-progress-v1'||payload.workspace_identity_sha256!==data.workspace_identity_sha256||!Array.isArray(payload.rows)||payload.rows.length!==data.packet.length)throw Error('Progress file belongs to a different workspace');$('reviewer').value=payload.reviewer_id||'';reviewer();const loaded=payload.rows.map((row,index)=>{{if(!row||row.review_identity_sha256!==data.packet[index].review_identity_sha256||typeof row.reviewed!=='boolean'||!row.draft||typeof row.draft!=='object'||Array.isArray(row.draft))throw Error('Progress row differs');if(row.reviewed)row.draft=normalize(row.draft,data.packet[index].text);return {{reviewed:row.reviewed,draft:row.draft}}}});states.splice(0,states.length,...loaded);current=0;render();message('Offline progress imported',true)}}catch(e){{message(e.message,false)}}finally{{$('progress-file').value=''}}}}
function render(){{$('heading').textContent=`Row ${{current+1}} / ${{data.packet.length}}`;$('source').textContent=data.packet[current].text;$('draft').value=JSON.stringify(states[current].draft,null,2);$('rows').replaceChildren(...states.map((state,index)=>{{const b=document.createElement('button');b.textContent=String(index+1);b.className=state.reviewed?'done':'';b.onclick=()=>{{try{{store(false)}}catch(e){{message(e.message,false);return}}current=index;render()}};return b}}));const done=states.filter(x=>x.reviewed).length;$('progress').textContent=`Reviewed ${{done}} / ${{states.length}}`;$('export').disabled=done!==states.length}}
function exportRows(){{try{{const rid=reviewer();store(false);if(states.some(x=>!x.reviewed))throw Error('Every row must be explicitly reviewed');const lines=states.map((state,index)=>JSON.stringify({{schema:data.review_row_schema,reviewer_id:rid,review_identity_sha256:data.packet[index].review_identity_sha256,...state.draft}})).join('\\n')+'\\n';download(`sai-finemath-human-review-${{rid}}.jsonl`,lines,'application/jsonl');message('Complete JSONL exported',true)}}catch(e){{message(e.message,false)}}}}
$('save').onclick=()=>{{try{{store(false)}}catch(e){{message(e.message,false)}}}};$('mark').onclick=()=>{{try{{store(true)}}catch(e){{message(e.message,false)}}}};$('backup').onclick=backup;$('restore').onclick=()=>$('progress-file').click();$('progress-file').onchange=event=>{{const file=event.target.files[0];if(file)restore(file)}};$('export').onclick=exportRows;render();
</script></body></html>"""
    return document.encode()


def _prepare(
    *, ladder_receipt: Path, expected_ladder_receipt_sha256: str
) -> tuple[dict[str, Any], bytes]:
    if not _HEX64.fullmatch(expected_ladder_receipt_sha256):
        raise FineMathReviewWorkspaceError("expected ladder receipt hash differs")
    if sha256_file(ladder_receipt) != expected_ladder_receipt_sha256:
        raise FineMathReviewWorkspaceError("expected ladder receipt hash differs")
    try:
        ladder = validate_ladder(ladder_receipt)
    except Exception as error:
        raise FineMathReviewWorkspaceError(
            "FineMath ladder validation failed"
        ) from error
    descriptor = ladder["blind_review_output"]
    packet_path = Path(descriptor["path"])
    rows, packet_encoded = _read_packet(
        packet_path, expected_rows=ladder["summary"]["blind_review_rows"]
    )
    if (
        hashlib.sha256(packet_encoded).hexdigest() != descriptor["sha256"]
        or len(packet_encoded) != descriptor["bytes"]
        or canonical_sha256(rows) != descriptor["ordered_rows_sha256"]
    ):
        raise FineMathReviewWorkspaceError("FineMath blind packet differs")
    identity = canonical_sha256(
        {
            "schema": SCHEMA,
            "ladder_receipt_sha256": ladder["receipt_sha256"],
            "ladder_receipt_file_sha256": expected_ladder_receipt_sha256,
            "blind_review_packet_sha256": descriptor["sha256"],
        }
    )
    html = _html_document(rows=rows, workspace_identity_sha256=identity)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "offline_blinded_human_quality_review_workspace",
        "ladder_receipt_sha256": ladder["receipt_sha256"],
        "ladder_receipt_file_sha256": expected_ladder_receipt_sha256,
        "blind_review_packet_sha256": descriptor["sha256"],
        "rows": len(rows),
        "workspace_identity_sha256": identity,
        "workspace_html": {
            "bytes": len(html),
            "sha256": hashlib.sha256(html).hexdigest(),
        },
        "offline_only": True,
        "external_requests": False,
        "hidden_review_key_included": False,
        "language_scores_included": False,
        "source_urls_included": False,
        "human_review_completed": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload, html


def build(
    *,
    ladder_receipt: Path,
    expected_ladder_receipt_sha256: str,
    workspace_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    if workspace_output.resolve() == receipt_output.resolve():
        raise FineMathReviewWorkspaceError("workspace outputs differ")
    payload, html = _prepare(
        ladder_receipt=ladder_receipt,
        expected_ladder_receipt_sha256=expected_ladder_receipt_sha256,
    )
    created = False
    try:
        _write_create_only(workspace_output, html)
        created = True
        _write_create_only(
            receipt_output,
            json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n",
        )
    except Exception as error:
        if created and not receipt_output.exists():
            workspace_output.chmod(0o600)
            workspace_output.unlink()
        raise FineMathReviewWorkspaceError(
            "workspace output boundary differs"
        ) from error
    return payload


def validate(
    *,
    ladder_receipt: Path,
    expected_ladder_receipt_sha256: str,
    workspace_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    expected, html = _prepare(
        ladder_receipt=ladder_receipt,
        expected_ladder_receipt_sha256=expected_ladder_receipt_sha256,
    )
    try:
        actual_html = _read_regular_bytes(workspace_output, maximum_bytes=32 << 20)
        actual = json.loads(_read_regular_bytes(receipt_output, maximum_bytes=1 << 20))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FineMathReviewWorkspaceError(
            "FineMath review workspace differs"
        ) from error
    if actual_html != html or actual != expected:
        raise FineMathReviewWorkspaceError("FineMath review workspace differs")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--ladder-receipt", type=Path, required=True)
    parser.add_argument("--expected-ladder-receipt-sha256", required=True)
    parser.add_argument("--workspace-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    payload = (build if command == "build" else validate)(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["rows"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

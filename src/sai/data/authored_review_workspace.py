"""Build an offline, blinded workspace for exact human curriculum review."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.authored_review_adjudication import (
    DEFECT_CATEGORIES,
    RECOMMENDATIONS,
)
from sai.data.authored_review_model import _blind_inputs
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-authored-curriculum-human-review-workspace-v1"


class AuthoredReviewWorkspaceError(RuntimeError):
    """The blinded packet, offline workspace, or receipt differs."""


def _embedded_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _html_document(
    *,
    packet: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    policy: dict[str, Any],
) -> bytes:
    data = {
        "packet": [
            {
                "review_identity_sha256": row["review_identity_sha256"],
                "text": row["text"],
            }
            for row in packet
        ],
        "concepts": [
            {
                "concept_id": row["concept_id"],
                "name": row["name"],
                "domain": row["domain"],
                "prerequisites": row["prerequisites"],
            }
            for row in concepts
        ],
        "minimum_confidence_ppm": policy["confidence_contract"][
            "minimum_confidence_ppm"
        ],
        "minimum_span_codepoints": policy["evidence_span_contract"][
            "minimum_codepoints_per_positive_label"
        ],
        "recommendations": sorted(RECOMMENDATIONS),
        "defect_categories": sorted(DEFECT_CATEGORIES - {"none"}),
    }
    encoded = _embedded_json(data)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:;">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sai blinded authored-curriculum review</title>
<style>
body{{margin:0;font:15px system-ui;color:#17202a;background:#f4f6f7}}header{{padding:12px 18px;background:#17202a;color:white}}main{{display:grid;grid-template-columns:220px 1fr 1fr;height:calc(100vh - 70px)}}aside,section{{overflow:auto;padding:12px;border-right:1px solid #ccd1d1}}button{{margin:2px;padding:6px 9px}}button.done{{background:#d5f5e3}}pre{{white-space:pre-wrap;word-break:break-word;background:white;padding:12px}}textarea{{width:100%;height:55vh;font:13px ui-monospace,monospace}}#concepts{{font-size:12px}}.error{{color:#922b21;font-weight:700}}.ok{{color:#196f3d;font-weight:700}}code{{background:#eaecee;padding:1px 3px}}
</style></head><body>
<header><strong>Sai blind human review</strong> — no curriculum phase, source identity, or provisional label is present. Select literal evidence directly from the chapter text.</header>
<main><aside><div id="progress"></div><div id="rows"></div></aside>
<section><h2 id="heading"></h2><pre id="source"></pre></section>
<section><p>Complete the JSON fields, then mark this row reviewed. Export is disabled until every row has been explicitly reviewed.</p><textarea id="draft"></textarea><div><button id="save">Save row</button><button id="mark">Mark reviewed</button><button id="export">Export complete JSONL</button></div><p id="message"></p><details><summary>Frozen concept vocabulary</summary><div id="concepts"></div></details></section></main>
<script id="sai-data" type="application/json">{encoded}</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('sai-data').textContent);
const conceptIds=new Set(data.concepts.map(x=>x.concept_id));
const blank=()=>({{instructional_quality_ppm:null,assumed_prior_concepts:[],taught_concepts:[],defects:[],admission_recommendation:null}});
const states=data.packet.map(()=>({{draft:blank(),reviewed:false}})); let current=0;
const $=id=>document.getElementById(id); const count=(text,part)=>text.split(part).length-1;
function normalize(value,text){{
 const keys=['admission_recommendation','assumed_prior_concepts','defects','instructional_quality_ppm','taught_concepts'];
 if(!value||typeof value!=='object'||Array.isArray(value)||Object.keys(value).sort().join('|')!==keys.join('|'))throw Error('Draft fields differ');
 if(!Number.isInteger(value.instructional_quality_ppm)||value.instructional_quality_ppm<0||value.instructional_quality_ppm>1000000)throw Error('Instructional quality must be an integer from 0 to 1000000');
 if(!data.recommendations.includes(value.admission_recommendation))throw Error('Choose an admission recommendation');
 for(const name of ['assumed_prior_concepts','taught_concepts','defects'])if(!Array.isArray(value[name]))throw Error(name+' must be an array');
 value.assumed_prior_concepts=[...new Set(value.assumed_prior_concepts)].sort();
 if(value.assumed_prior_concepts.some(x=>!conceptIds.has(x)))throw Error('Unknown assumed concept');
 const seen=new Set();
 for(const item of value.taught_concepts){{
  if(!item||Object.keys(item).sort().join('|')!=='concept_id|confidence_ppm|evidence_quotes')throw Error('Taught evidence fields differ');
  if(!conceptIds.has(item.concept_id)||seen.has(item.concept_id))throw Error('Unknown or duplicate taught concept'); seen.add(item.concept_id);
  if(!Number.isInteger(item.confidence_ppm)||item.confidence_ppm<data.minimum_confidence_ppm||item.confidence_ppm>1000000)throw Error('Taught confidence is below the frozen floor');
  if(!Array.isArray(item.evidence_quotes)||!item.evidence_quotes.length)throw Error('Taught evidence requires a quote');
  for(const quote of item.evidence_quotes)if(typeof quote!=='string'||[...quote].length<data.minimum_span_codepoints||count(text,quote)!==1)throw Error('Every taught quote must be a unique literal source span');
 }}
 value.taught_concepts.sort((a,b)=>a.concept_id.localeCompare(b.concept_id));
 if(value.assumed_prior_concepts.some(x=>seen.has(x)))throw Error('A concept cannot be both assumed and taught');
 if(value.admission_recommendation==='admit'&&!value.taught_concepts.length)throw Error('Admit requires evidence-backed teaching');
 for(const item of value.defects){{
  if(!item||Object.keys(item).sort().join('|')!=='category|evidence_quote'||!data.defect_categories.includes(item.category))throw Error('Defect fields differ');
  const quote=item.evidence_quote;if(typeof quote!=='string'||[...quote].length<data.minimum_span_codepoints||count(text,quote)!==1)throw Error('Every defect quote must be a unique literal source span');
 }}
 return value;
}}
function store(mark){{const row=data.packet[current];const value=normalize(JSON.parse($('draft').value),row.text);states[current].draft=value;if(mark)states[current].reviewed=true;render();message(mark?'Row marked reviewed':'Row saved',true)}}
function message(text,ok){{$('message').textContent=text;$('message').className=ok?'ok':'error'}}
function render(){{
 $('heading').textContent=`Row ${{current+1}} / ${{data.packet.length}}`; $('source').textContent=data.packet[current].text;$('draft').value=JSON.stringify(states[current].draft,null,2);
 $('rows').replaceChildren(...states.map((state,index)=>{{const b=document.createElement('button');b.textContent=String(index+1);b.className=state.reviewed?'done':'';b.onclick=()=>{{try{{store(false)}}catch(e){{message(e.message,false);return}}current=index;render()}};return b}}));
 const done=states.filter(x=>x.reviewed).length;$('progress').textContent=`Reviewed ${{done}} / ${{states.length}}`;$('export').disabled=done!==states.length;
}}
function exportRows(){{try{{store(false);if(states.some(x=>!x.reviewed))throw Error('Every row must be explicitly reviewed');const lines=states.map((state,index)=>JSON.stringify({{schema:'sai-authored-curriculum-quoted-review-draft-row-v1',review_identity_sha256:data.packet[index].review_identity_sha256,...state.draft}})).join('\\n')+'\\n';const blob=new Blob([lines],{{type:'application/jsonl'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='sai-authored-human-review.jsonl';a.click();URL.revokeObjectURL(a.href);message('Complete JSONL exported',true)}}catch(e){{message(e.message,false)}}}}
$('save').onclick=()=>{{try{{store(false)}}catch(e){{message(e.message,false)}}}};$('mark').onclick=()=>{{try{{store(true)}}catch(e){{message(e.message,false)}}}};$('export').onclick=exportRows;
$('concepts').innerHTML=data.concepts.map(x=>`<p><code>${{x.concept_id}}</code> — ${{x.name}} (${{x.domain}}); prerequisites: ${{x.prerequisites.join(', ')||'none'}}</p>`).join('');render();
</script></body></html>"""
    return document.encode()


def _prepare(
    *,
    review_packet: Path,
    review_packet_receipt: Path,
    expected_review_packet_sha256: str,
    expected_review_packet_receipt_sha256: str,
    concept_list: Path,
    annotation_policy: Path,
) -> tuple[dict[str, Any], bytes]:
    try:
        inputs = _blind_inputs(
            review_packet=review_packet,
            review_packet_receipt=review_packet_receipt,
            expected_review_packet_sha256=expected_review_packet_sha256,
            expected_review_packet_receipt_sha256=expected_review_packet_receipt_sha256,
            concept_list=concept_list,
            annotation_policy=annotation_policy,
        )
    except Exception as error:
        raise AuthoredReviewWorkspaceError("blind review inputs differ") from error
    encoded = _html_document(
        packet=inputs.packet,
        concepts=inputs.concept_payload["concepts"],
        policy=inputs.policy,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "offline_blinded_human_review_workspace",
        "blind_review_packet_sha256": hashlib.sha256(inputs.packet_encoded).hexdigest(),
        "blind_review_packet_receipt_sha256": hashlib.sha256(
            inputs.packet_receipt_encoded
        ).hexdigest(),
        "concept_list_sha256": hashlib.sha256(inputs.concept_encoded).hexdigest(),
        "annotation_policy_sha256": hashlib.sha256(inputs.policy_encoded).hexdigest(),
        "rows": len(inputs.packet),
        "workspace_html": {
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        },
        "offline_only": True,
        "external_requests": False,
        "hidden_review_key_included": False,
        "provisional_curriculum_labels_included": False,
        "human_review_completed": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload, encoded


def build(
    *, workspace_output: Path, receipt_output: Path, **kwargs: Any
) -> dict[str, Any]:
    if workspace_output.resolve() == receipt_output.resolve():
        raise AuthoredReviewWorkspaceError("workspace outputs differ")
    payload, encoded = _prepare(**kwargs)
    created_workspace = False
    try:
        _write_create_only(workspace_output, encoded)
        created_workspace = True
        _write_create_only(
            receipt_output,
            json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n",
        )
    except Exception as error:
        if created_workspace and not receipt_output.exists():
            workspace_output.chmod(0o600)
            workspace_output.unlink()
        raise AuthoredReviewWorkspaceError(
            "workspace output boundary differs"
        ) from error
    return payload


def validate(
    *, workspace_output: Path, receipt_output: Path, **kwargs: Any
) -> dict[str, Any]:
    expected, encoded = _prepare(**kwargs)
    try:
        workspace = _read_regular_bytes(workspace_output, maximum_bytes=64 << 20)
        actual = json.loads(_read_regular_bytes(receipt_output, maximum_bytes=1 << 20))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredReviewWorkspaceError("human review workspace differs") from error
    if workspace != encoded or actual != expected:
        raise AuthoredReviewWorkspaceError("human review workspace differs")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    for name in (
        "review-packet",
        "review-packet-receipt",
        "concept-list",
        "annotation-policy",
        "workspace-output",
        "receipt-output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-review-packet-sha256", required=True)
    parser.add_argument("--expected-review-packet-receipt-sha256", required=True)
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

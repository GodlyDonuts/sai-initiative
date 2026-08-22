"""Build an offline subject-review workspace for the Sai syllabus graph."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.curriculum import PHASES
from sai.data.foundational_syllabus import _prepare as _prepare_syllabus
from sai.data.foundational_syllabus_audit import _prepare as _prepare_audit
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-foundational-syllabus-review-workspace-v1"
REVIEW_ROW_SCHEMA = "sai-foundational-syllabus-subject-review-v1"


class FoundationalSyllabusReviewWorkspaceError(RuntimeError):
    """The composed graph, review workspace, or receipt differs."""


def _embedded_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _earliest_phase(row: dict[str, Any]) -> str:
    return next(phase for phase in PHASES if row["minimum_phase_documents"][phase] > 0)


def _review_rows(
    concept_payload: dict[str, Any], audit_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    concepts = concept_payload["concepts"]
    by_identity = {row["concept_id"]: row for row in concepts}
    graph_rows = {row["concept_id"]: row for row in audit_payload["concept_graph_rows"]}
    dependents: dict[str, list[str]] = defaultdict(list)
    for row in concepts:
        for prerequisite in row["prerequisites"]:
            dependents[prerequisite].append(row["concept_id"])
    rows = []
    for concept in concepts:
        rows.append(
            {
                "concept_id": concept["concept_id"],
                "name": concept["name"],
                "domain": concept["domain"],
                "earliest_phase": _earliest_phase(concept),
                "minimum_prior_documents": concept["minimum_prior_documents"],
                "prerequisites": [
                    {
                        "concept_id": prerequisite,
                        "name": by_identity[prerequisite]["name"],
                        "domain": by_identity[prerequisite]["domain"],
                        "earliest_phase": _earliest_phase(by_identity[prerequisite]),
                    }
                    for prerequisite in concept["prerequisites"]
                ],
                "direct_dependents": sorted(dependents[concept["concept_id"]]),
                "hard_prerequisite_depth": graph_rows[concept["concept_id"]][
                    "hard_prerequisite_depth"
                ],
                "transitive_hard_prerequisites": graph_rows[concept["concept_id"]][
                    "transitive_hard_prerequisites"
                ],
                "risk_flags": graph_rows[concept["concept_id"]]["risk_flags"],
            }
        )
    return rows


def _html_document(
    *, review_rows: list[dict[str, Any]], workspace_identity_sha256: str
) -> bytes:
    data = {
        "workspace_identity_sha256": workspace_identity_sha256,
        "review_row_schema": REVIEW_ROW_SCHEMA,
        "phases": list(PHASES),
        "concept_ids": [row["concept_id"] for row in review_rows],
        "rows": review_rows,
    }
    encoded = _embedded_json(data)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:;">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sai foundational syllabus subject review</title>
<style>
body{{margin:0;font:15px system-ui;color:#17202a;background:#f4f6f7}}header{{padding:12px 18px;background:#17202a;color:white}}main{{display:grid;grid-template-columns:240px 1fr 1fr;height:calc(100vh - 72px)}}aside,section{{overflow:auto;padding:12px;border-right:1px solid #ccd1d1}}button{{margin:2px;padding:6px 9px}}button.done{{background:#d5f5e3}}pre{{white-space:pre-wrap;word-break:break-word;background:white;padding:12px}}textarea{{width:100%;height:52vh;font:13px ui-monospace,monospace}}input{{padding:5px}}.error{{color:#922b21;font-weight:700}}.ok{{color:#196f3d;font-weight:700}}.risk{{color:#922b21}}code{{background:#eaecee;padding:1px 3px}}
</style></head><body>
<header><strong>Sai syllabus subject review</strong> — classify every declared dependency as hard, supporting, or removable before document annotation.</header>
<main><aside><label>Reviewer pseudonym <input id="reviewer" autocomplete="off" pattern="[A-Za-z0-9_-]{{3,64}}"></label><div id="progress"></div><div id="rows"></div></aside>
<section><h2 id="heading"></h2><div id="graph"></div></section>
<section><p>Review the concept independently from its current graph status. Every existing edge requires a rationale. Add only prerequisite concepts that already exist in the frozen inventory.</p><textarea id="draft"></textarea><div><button id="save">Save row</button><button id="mark">Mark reviewed</button><button id="backup">Export progress</button><button id="restore">Import progress</button><input id="progress-file" type="file" accept="application/json,.json" hidden><button id="export">Export complete JSONL</button></div><p id="message"></p></section></main>
<script id="sai-data" type="application/json">{encoded}</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('sai-data').textContent);const conceptIds=new Set(data.concept_ids);
const blank=row=>({{concept_verdict:null,proposed_name:null,proposed_earliest_phase:row.earliest_phase,granularity:null,edge_reviews:row.prerequisites.map(x=>({{prerequisite_id:x.concept_id,classification:null,rationale:''}})),missing_prerequisites:[],rationale:''}});
const states=data.rows.map(row=>({{draft:blank(row),reviewed:false}}));let current=0;const $=id=>document.getElementById(id);
function reviewer(){{const value=$('reviewer').value.trim();if(!/^[A-Za-z0-9_-]{{3,64}}$/.test(value))throw Error('Reviewer pseudonym must be 3-64 letters, digits, underscores, or hyphens');return value}}
function normalize(value,row){{
 const keys=['concept_verdict','edge_reviews','granularity','missing_prerequisites','proposed_earliest_phase','proposed_name','rationale'];if(!value||typeof value!=='object'||Array.isArray(value)||Object.keys(value).sort().join('|')!==keys.join('|'))throw Error('Draft fields differ');
 if(!['accept','revise','reject'].includes(value.concept_verdict)||!['appropriate','too_broad','too_narrow'].includes(value.granularity)||!data.phases.includes(value.proposed_earliest_phase))throw Error('Choose valid concept decisions');
 if(value.proposed_name!==null&&(typeof value.proposed_name!=='string'||value.proposed_name.trim().length<3))throw Error('Proposed name differs');if(typeof value.rationale!=='string'||value.rationale.trim().length<40)throw Error('Concept rationale must contain at least 40 characters');
 if(!Array.isArray(value.edge_reviews)||value.edge_reviews.length!==row.prerequisites.length)throw Error('Every existing edge must be reviewed');const expected=row.prerequisites.map(x=>x.concept_id).sort();const observed=value.edge_reviews.map(x=>x&&x.prerequisite_id).sort();if(expected.join('|')!==observed.join('|'))throw Error('Existing edge identities differ');
 for(const edge of value.edge_reviews)if(!edge||Object.keys(edge).sort().join('|')!=='classification|prerequisite_id|rationale'||!['hard','supporting','remove'].includes(edge.classification)||typeof edge.rationale!=='string'||edge.rationale.trim().length<20)throw Error('Every edge needs a classification and 20-character rationale');value.edge_reviews.sort((a,b)=>a.prerequisite_id.localeCompare(b.prerequisite_id));
 if(!Array.isArray(value.missing_prerequisites))throw Error('missing_prerequisites must be an array');const currentPrereqs=new Set(expected);const missingSeen=new Set();for(const edge of value.missing_prerequisites){{if(!edge||Object.keys(edge).sort().join('|')!=='classification|prerequisite_id|rationale'||!conceptIds.has(edge.prerequisite_id)||edge.prerequisite_id===row.concept_id||currentPrereqs.has(edge.prerequisite_id)||missingSeen.has(edge.prerequisite_id)||!['hard','supporting'].includes(edge.classification)||typeof edge.rationale!=='string'||edge.rationale.trim().length<20)throw Error('Proposed prerequisite differs');missingSeen.add(edge.prerequisite_id)}}value.missing_prerequisites.sort((a,b)=>a.prerequisite_id.localeCompare(b.prerequisite_id));
 if(value.concept_verdict==='accept'&&(value.proposed_name!==null||value.proposed_earliest_phase!==row.earliest_phase||value.granularity!=='appropriate'))throw Error('Accept must preserve name, phase, and granularity');if(value.concept_verdict==='revise'&&value.proposed_name===null&&value.proposed_earliest_phase===row.earliest_phase&&value.granularity==='appropriate'&&value.edge_reviews.every(x=>x.classification==='hard')&&!value.missing_prerequisites.length)throw Error('Revise must declare a material change');return value;
}}
function store(mark){{const row=data.rows[current];const value=normalize(JSON.parse($('draft').value),row);states[current].draft=value;if(mark)states[current].reviewed=true;render();message(mark?'Row marked reviewed':'Row saved',true)}}function message(text,ok){{$('message').textContent=text;$('message').className=ok?'ok':'error'}}
function download(name,text,type){{const blob=new Blob([text],{{type}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href)}}
function backup(){{try{{const rid=reviewer();states[current].draft=JSON.parse($('draft').value);const payload={{schema:'sai-foundational-syllabus-review-progress-v1',workspace_identity_sha256:data.workspace_identity_sha256,reviewer_id:rid,rows:states.map((state,index)=>({{concept_id:data.rows[index].concept_id,reviewed:state.reviewed,draft:state.draft}}))}};download('sai-foundational-syllabus-review-progress.json',JSON.stringify(payload,null,2)+'\\n','application/json');message('Offline progress exported',true)}}catch(e){{message(e.message,false)}}}}
async function restore(file){{try{{const payload=JSON.parse(await file.text());if(!payload||payload.schema!=='sai-foundational-syllabus-review-progress-v1'||payload.workspace_identity_sha256!==data.workspace_identity_sha256||!Array.isArray(payload.rows)||payload.rows.length!==data.rows.length)throw Error('Progress file belongs to a different workspace');$('reviewer').value=payload.reviewer_id||'';reviewer();const loaded=payload.rows.map((entry,index)=>{{if(!entry||entry.concept_id!==data.rows[index].concept_id||typeof entry.reviewed!=='boolean'||!entry.draft||typeof entry.draft!=='object'||Array.isArray(entry.draft))throw Error('Progress row differs');if(entry.reviewed)entry.draft=normalize(entry.draft,data.rows[index]);return {{reviewed:entry.reviewed,draft:entry.draft}}}});states.splice(0,states.length,...loaded);current=0;render();message('Offline progress imported',true)}}catch(e){{message(e.message,false)}}finally{{$('progress-file').value=''}}}}
function render(){{const row=data.rows[current];$('heading').textContent=`${{row.concept_id}} — ${{row.name}}`;$('graph').innerHTML=`<p>Domain: <code>${{row.domain}}</code>; earliest phase: <code>${{row.earliest_phase}}</code>; hard depth: ${{row.hard_prerequisite_depth}}; transitive prerequisites: ${{row.transitive_hard_prerequisites}}</p><p class="risk">Risk flags: ${{row.risk_flags.join(', ')||'none'}}</p><h3>Current prerequisites</h3>${{row.prerequisites.map(x=>`<p><code>${{x.concept_id}}</code> — ${{x.name}} (${{x.domain}}, ${{x.earliest_phase}})</p>`).join('')||'<p>none</p>'}}<h3>Direct dependents</h3><p>${{row.direct_dependents.map(x=>`<code>${{x}}</code>`).join(' ')||'none'}}</p>`;$('draft').value=JSON.stringify(states[current].draft,null,2);$('rows').replaceChildren(...states.map((state,index)=>{{const b=document.createElement('button');b.textContent=String(index+1);b.title=data.rows[index].concept_id;b.className=state.reviewed?'done':'';b.onclick=()=>{{try{{store(false)}}catch(e){{message(e.message,false);return}}current=index;render()}};return b}}));const done=states.filter(x=>x.reviewed).length;$('progress').textContent=`Reviewed ${{done}} / ${{states.length}}`;$('export').disabled=done!==states.length}}
function exportRows(){{try{{const rid=reviewer();store(false);if(states.some(x=>!x.reviewed))throw Error('Every concept must be explicitly reviewed');const lines=states.map((state,index)=>JSON.stringify({{schema:data.review_row_schema,reviewer_id:rid,concept_id:data.rows[index].concept_id,...state.draft}})).join('\\n')+'\\n';download(`sai-foundational-syllabus-review-${{rid}}.jsonl`,lines,'application/jsonl');message('Complete JSONL exported',true)}}catch(e){{message(e.message,false)}}}}
$('save').onclick=()=>{{try{{store(false)}}catch(e){{message(e.message,false)}}}};$('mark').onclick=()=>{{try{{store(true)}}catch(e){{message(e.message,false)}}}};$('backup').onclick=backup;$('restore').onclick=()=>$('progress-file').click();$('progress-file').onchange=event=>{{const file=event.target.files[0];if(file)restore(file)}};$('export').onclick=exportRows;render();
</script></body></html>"""
    return document.encode()


def _prepare(*, base_concepts: Path, additions: Path) -> tuple[dict[str, Any], bytes]:
    try:
        composition_receipt, concept_encoded, _ = _prepare_syllabus(
            base_concepts=base_concepts, additions=additions
        )
        audit_payload, audit_encoded = _prepare_audit(
            base_concepts=base_concepts, additions=additions
        )
        concept_payload = json.loads(concept_encoded)
    except Exception as error:
        raise FoundationalSyllabusReviewWorkspaceError(
            "syllabus review inputs differ"
        ) from error
    rows = _review_rows(concept_payload, audit_payload)
    identity = canonical_sha256(
        {
            "schema": SCHEMA,
            "composition_receipt_sha256": composition_receipt["receipt_sha256"],
            "concept_sha256": hashlib.sha256(concept_encoded).hexdigest(),
            "audit_receipt_sha256": audit_payload["receipt_sha256"],
            "audit_file_sha256": hashlib.sha256(audit_encoded).hexdigest(),
            "review_rows_sha256": canonical_sha256(rows),
        }
    )
    html = _html_document(review_rows=rows, workspace_identity_sha256=identity)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "offline_subject_review_workspace",
        "composition_receipt_sha256": composition_receipt["receipt_sha256"],
        "concept_sha256": hashlib.sha256(concept_encoded).hexdigest(),
        "audit_receipt_sha256": audit_payload["receipt_sha256"],
        "review_rows_sha256": canonical_sha256(rows),
        "concepts": len(rows),
        "hard_edges": sum(len(row["prerequisites"]) for row in rows),
        "flagged_concepts": sum(bool(row["risk_flags"]) for row in rows),
        "workspace_identity_sha256": identity,
        "workspace_html": {
            "bytes": len(html),
            "sha256": hashlib.sha256(html).hexdigest(),
        },
        "offline_only": True,
        "external_requests": False,
        "subject_review_completed": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload, html


def build(
    *,
    base_concepts: Path,
    additions: Path,
    workspace_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    if workspace_output.resolve() == receipt_output.resolve():
        raise FoundationalSyllabusReviewWorkspaceError("workspace outputs differ")
    payload, html = _prepare(base_concepts=base_concepts, additions=additions)
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
        raise FoundationalSyllabusReviewWorkspaceError(
            "workspace output boundary differs"
        ) from error
    return payload


def validate(
    *,
    base_concepts: Path,
    additions: Path,
    workspace_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    payload, html = _prepare(base_concepts=base_concepts, additions=additions)
    try:
        actual_html = _read_regular_bytes(workspace_output, maximum_bytes=8 << 20)
        actual_receipt = json.loads(
            _read_regular_bytes(receipt_output, maximum_bytes=1 << 20)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FoundationalSyllabusReviewWorkspaceError(
            "syllabus review workspace differs"
        ) from error
    if actual_html != html or actual_receipt != payload:
        raise FoundationalSyllabusReviewWorkspaceError(
            "syllabus review workspace differs"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--base-concepts", type=Path, required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--workspace-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    payload = (build if command == "build" else validate)(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "concepts": payload["concepts"],
                "hard_edges": payload["hard_edges"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

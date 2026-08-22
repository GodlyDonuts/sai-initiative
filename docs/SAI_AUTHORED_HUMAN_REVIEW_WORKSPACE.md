# Sai blinded authored-curriculum human-review workspace

`sai-build-authored-review-workspace` converts the already-frozen 127-row blind
packet into one self-contained offline HTML workspace. It exists to make the
required independent human review practical without exposing the hidden
curriculum key or asking reviewers to edit JSONL by hand.

The workspace contains only each salted review identity, the exact chapter
text, the frozen candidate concept vocabulary, and the frozen evidence rules.
It contains no source identity, publisher order, proposed phase, surface band,
model-generated label, or provisional admission decision. Its content-security
policy permits only embedded style and script; the page performs no network
request. Browser state starts unlabeled. Every row must be explicitly marked
reviewed before the page exports the exact quoted-review JSONL schema consumed
by `sai-compile-authored-review`.

The browser checks the frozen confidence floor, concept vocabulary, role
disjointness, and unique literal evidence spans before marking a row complete.
The Python compiler remains authoritative and reopens every quote against the
immutable packet. Two separately identified humans must independently export,
compile, and attest all 127 rows. The workspace receipt itself records
`human_review_completed=false`, `training_authorized=false`, and
`four_b_training_authorized=false`; generating a convenient form is not data
qualification.

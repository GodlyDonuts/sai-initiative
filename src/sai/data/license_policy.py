"""Normalize exact source license declarations without inferring missing rights."""

from __future__ import annotations

from typing import Any

from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-exact-declared-license-policy-v1"
POLICY = {
    "normalization": "strip_casefold_exact_alias_match",
    "unknown_or_ambiguous_action": "rights_hold",
    "public_domain_is_a_declaration_not_jurisdictional_verification": True,
    "license_declaration_does_not_verify_source_provenance": True,
}

_LICENSES = {
    "apache-2.0": ("Apache-2.0", True, False),
    "bsd-2-clause": ("BSD-2-Clause", True, False),
    "cc0-1.0": ("CC0-1.0", False, False),
    "cc-by-3.0": ("CC-BY-3.0", True, False),
    "cc-by-4.0": ("CC-BY-4.0", True, False),
    "cc-by-sa-2.5": ("CC-BY-SA-2.5", True, True),
    "cc-by-sa-3.0": ("CC-BY-SA-3.0", True, True),
    "cc-by-sa-4.0": ("CC-BY-SA-4.0", True, True),
    "mit": ("MIT", True, False),
    "wtfpl": ("WTFPL", False, False),
    "public domain": ("LicenseRef-Public-Domain", False, False),
    (
        "creative commons zero - public domain - "
        "https://creativecommons.org/publicdomain/zero/1.0/"
    ): ("CC0-1.0", False, False),
    (
        "creative commons - attribution - "
        "https://creativecommons.org/licenses/by/3.0/"
    ): ("CC-BY-3.0", True, False),
    (
        "creative commons - attribution - "
        "https://creativecommons.org/licenses/by/4.0/"
    ): ("CC-BY-4.0", True, False),
    (
        "creative commons - attribution share-alike - "
        "https://creativecommons.org/licenses/by-sa/2.5/"
    ): ("CC-BY-SA-2.5", True, True),
    (
        "creative commons - attribution share-alike - "
        "https://creativecommons.org/licenses/by-sa/3.0/"
    ): ("CC-BY-SA-3.0", True, True),
    (
        "creative commons - attribution share-alike - "
        "https://creativecommons.org/licenses/by-sa/4.0/"
    ): ("CC-BY-SA-4.0", True, True),
}


class LicensePolicyError(RuntimeError):
    """A declared license value or frozen alias policy differs."""


def classify_declared_license(value: Any) -> dict[str, Any]:
    """Return a source-safe declaration classification, never a legal conclusion."""

    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise LicensePolicyError("declared license differs")
    declared = value.strip()
    normalized = " ".join(declared.casefold().split())
    match = _LICENSES.get(normalized)
    if match is None:
        result = {
            "schema": SCHEMA,
            "declared_license": declared,
            "canonical_license": None,
            "declaration_recognized": False,
            "rights_hold": True,
            "attribution_required": None,
            "share_alike_required": None,
            "source_provenance_verified": False,
            "legal_clearance_established": False,
        }
    else:
        canonical, attribution, share_alike = match
        result = {
            "schema": SCHEMA,
            "declared_license": declared,
            "canonical_license": canonical,
            "declaration_recognized": True,
            "rights_hold": False,
            "attribution_required": attribution,
            "share_alike_required": share_alike,
            "source_provenance_verified": False,
            "legal_clearance_established": False,
        }
    result["classification_sha256"] = canonical_sha256(result)
    return result

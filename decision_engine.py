
import hashlib
import json
import uuid
import re

VALID_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}


def canonical_json(data):
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )


def content_hash(data):
    return hashlib.sha256(
        canonical_json(data).encode("utf-8")
    ).hexdigest()


def create_action_id():
    return str(uuid.uuid4())


def validate_decision(decision, allowed_evidence_refs=None):
    if not isinstance(decision, dict):
        raise ValueError("Decision must be an object")

    action = decision.get("action")

    if action not in VALID_ACTIONS:
        raise ValueError("Invalid invoice action")

    facts = decision.get("facts")

    if not isinstance(facts, dict):
        raise ValueError("Decision facts are missing")

    required_facts = [
        "vendorName",
        "invoiceNumber",
        "amountMinor",
        "currency",
    ]

    for field in required_facts:
        if field not in facts:
            raise ValueError(f"Missing fact: {field}")

    for field in ["vendorName", "invoiceNumber"]:
        if not isinstance(facts[field], str) or not facts[field].strip():
            raise ValueError(f"Invalid fact: {field}")

    amount = facts["amountMinor"]

    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        raise ValueError("amountMinor must be a non-negative integer")

    currency = facts["currency"]

    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("Currency must be a three-letter uppercase code")

    evidence = decision.get("evidenceRefs")

    if not isinstance(evidence, list) or len(evidence) != 3:
        raise ValueError("Exactly three evidence references are required")

    if not all(isinstance(ref, str) and ref.strip() for ref in evidence):
        raise ValueError("Evidence references must be non-empty strings")

    if len(set(evidence)) != 3:
        raise ValueError("Evidence references must be unique")

    if allowed_evidence_refs is not None:
        if not set(evidence).issubset(set(allowed_evidence_refs)):
            raise ValueError("Evidence reference does not exist in the package")

    rationale = decision.get("rationale")

    if not isinstance(rationale, str) or not 60 <= len(rationale) <= 1500:
        raise ValueError("Rationale must contain 60–1500 characters")

    if action not in rationale:
        raise ValueError("Rationale must name the selected action")

    cited_refs = [ref for ref in evidence if ref in rationale]

    if len(cited_refs) < 2:
        raise ValueError("Rationale must cite at least two evidence references")

    return decision


def create_proposal(package_id, decision, allowed_evidence_refs=None):
    if not isinstance(package_id, str) or not package_id.strip():
        raise ValueError("Invalid package ID")

    validate_decision(decision, allowed_evidence_refs)

    return {
        "packageId": package_id,
        "actionId": create_action_id(),
        "action": decision["action"],
        "facts": decision["facts"],
        "evidenceRefs": decision["evidenceRefs"],
        "rationale": decision["rationale"],
    }

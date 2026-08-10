"""Fail-closed contract for the convertible-bond benchmark dataset.

Validates provenance and layer separation. It never infers missing legal terms,
market observations, fair values, or recommendations.
"""
from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "bmax.convertible-benchmark.v1"
PROTECTION_STATES = {"PRESENT", "ABSENT", "UNVERIFIED"}
REQUIRED_ISSUE_FIELDS = (
    "principal", "issue_date", "maturity_date", "coupon", "currency",
    "conversion_terms.conversion_price", "conversion_terms.conversion_ratio",
    "conversion_terms.reference_share",
)
PROTECTION_FIELDS = ("call_terms", "put_terms", "redemption_terms", "fundamental_change_terms")
FORBIDDEN_CONTRACT_KEYS = {
    "equity_price", "risk_free_curve_ref", "credit_spread", "bond_floor",
    "conversion_value", "option_component", "var", "expected_shortfall",
    "fair_value", "recommendation",
}


class BenchmarkContractError(ValueError):
    pass


def _obj(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkContractError(f"{field} must be an array")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkContractError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkContractError(f"{field} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise BenchmarkContractError(f"{field} must be >= {minimum}")
    return result


def _date(value: Any, field: str) -> str:
    text = _text(value, field)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise BenchmarkContractError(f"{field} must be ISO date YYYY-MM-DD") from exc
    return text


def _https_url(value: Any, field: str) -> str:
    text = _text(value, field)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise BenchmarkContractError(f"{field} must be an https URL")
    return text


def _records(records: list[Any], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for idx, raw in enumerate(records):
        record = _obj(raw, f"{field}[{idx}]")
        record_id = _text(record.get("id"), f"{field}[{idx}].id")
        if record_id in result:
            raise BenchmarkContractError(f"duplicate {field} id: {record_id}")
        result[record_id] = record
    return result


def _nested(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise BenchmarkContractError(f"{record.get('id', '<record>')}: missing {path}")
        current = current[part]
    return current


def validate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    root = _obj(dataset, "dataset")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkContractError(f"schema_version must be {SCHEMA_VERSION}")

    issuers = _records(_list(root.get("issuers"), "issuers"), "issuers")
    issues = _records(_list(root.get("issues"), "issues"), "issues")
    evidence = _records(_list(root.get("evidence"), "evidence"), "evidence")
    snapshots = _records(_list(root.get("market_snapshots", []), "market_snapshots"), "market_snapshots")
    scenarios = _records(_list(root.get("scenarios", []), "scenarios"), "scenarios")

    for issuer_id, issuer in issuers.items():
        _text(issuer.get("legal_name"), f"{issuer_id}.legal_name")
        _text(issuer.get("ticker"), f"{issuer_id}.ticker")
        cik = _text(issuer.get("cik"), f"{issuer_id}.cik")
        if not cik.isdigit():
            raise BenchmarkContractError(f"{issuer_id}.cik must contain digits only")
        _text(issuer.get("jurisdiction"), f"{issuer_id}.jurisdiction")

    for evidence_id, item in evidence.items():
        _text(item.get("accession_no"), f"{evidence_id}.accession_no")
        _text(item.get("filing_type"), f"{evidence_id}.filing_type")
        _date(item.get("filed_at"), f"{evidence_id}.filed_at")
        _https_url(item.get("source_url"), f"{evidence_id}.source_url")
        _text(item.get("section_ref"), f"{evidence_id}.section_ref")
        _date(item.get("observed_at"), f"{evidence_id}.observed_at")

    evidence_complete = 0
    scenario_ready = 0
    for issue_id, issue in issues.items():
        issuer_id = _text(issue.get("issuer_id"), f"{issue_id}.issuer_id")
        if issuer_id not in issuers:
            raise BenchmarkContractError(f"{issue_id}: unknown issuer_id {issuer_id}")
        overlap = FORBIDDEN_CONTRACT_KEYS.intersection(issue)
        if overlap:
            raise BenchmarkContractError(f"{issue_id}: market/model fields forbidden in contract layer: {sorted(overlap)}")

        _number(issue.get("principal"), f"{issue_id}.principal", minimum=0.01)
        issue_date = _date(issue.get("issue_date"), f"{issue_id}.issue_date")
        maturity_date = _date(issue.get("maturity_date"), f"{issue_id}.maturity_date")
        if maturity_date <= issue_date:
            raise BenchmarkContractError(f"{issue_id}: maturity_date must be after issue_date")
        coupon = _number(issue.get("coupon"), f"{issue_id}.coupon", minimum=0.0)
        if coupon > 1:
            raise BenchmarkContractError(f"{issue_id}.coupon must be a decimal rate <= 1")
        _text(issue.get("currency"), f"{issue_id}.currency")

        conversion = _obj(issue.get("conversion_terms"), f"{issue_id}.conversion_terms")
        _number(conversion.get("conversion_price"), f"{issue_id}.conversion_price", minimum=0.01)
        _number(conversion.get("conversion_ratio"), f"{issue_id}.conversion_ratio", minimum=0.000001)
        _text(conversion.get("reference_share"), f"{issue_id}.reference_share")

        protection = _obj(issue.get("protection_terms"), f"{issue_id}.protection_terms")
        all_protection_verified = True
        for name in PROTECTION_FIELDS:
            term = _obj(protection.get(name), f"{issue_id}.{name}")
            state = _text(term.get("status"), f"{issue_id}.{name}.status")
            if state not in PROTECTION_STATES:
                raise BenchmarkContractError(f"{issue_id}.{name}.status unsupported: {state}")
            evidence_id = term.get("evidence_id")
            if state == "UNVERIFIED":
                all_protection_verified = False
                if evidence_id not in (None, ""):
                    raise BenchmarkContractError(f"{issue_id}.{name}: UNVERIFIED must not cite confirming evidence")
            elif _text(evidence_id, f"{issue_id}.{name}.evidence_id") not in evidence:
                raise BenchmarkContractError(f"{issue_id}.{name}: unknown evidence_id")

        evidence_map = _obj(issue.get("evidence_map"), f"{issue_id}.evidence_map")
        missing_evidence = []
        for field in REQUIRED_ISSUE_FIELDS:
            _nested(issue, field)
            evidence_id = evidence_map.get(field)
            if not isinstance(evidence_id, str) or evidence_id not in evidence:
                missing_evidence.append(field)
        if not missing_evidence:
            evidence_complete += 1
        if not missing_evidence and all_protection_verified:
            scenario_ready += 1

    for snapshot_id, snapshot in snapshots.items():
        issue_id = _text(snapshot.get("issue_id"), f"{snapshot_id}.issue_id")
        if issue_id not in issues:
            raise BenchmarkContractError(f"{snapshot_id}: unknown issue_id {issue_id}")
        _date(snapshot.get("as_of"), f"{snapshot_id}.as_of")
        _number(snapshot.get("equity_price"), f"{snapshot_id}.equity_price", minimum=0.01)
        _text(snapshot.get("risk_free_curve_ref"), f"{snapshot_id}.risk_free_curve_ref")
        if snapshot.get("credit_spread_if_observed") is not None:
            _number(snapshot["credit_spread_if_observed"], f"{snapshot_id}.credit_spread_if_observed", minimum=0)

    for scenario_id, scenario in scenarios.items():
        issue_id = _text(scenario.get("issue_id"), f"{scenario_id}.issue_id")
        snapshot_id = _text(scenario.get("snapshot_id"), f"{scenario_id}.snapshot_id")
        if issue_id not in issues or snapshot_id not in snapshots:
            raise BenchmarkContractError(f"{scenario_id}: unknown issue/snapshot reference")
        if snapshots[snapshot_id].get("issue_id") != issue_id:
            raise BenchmarkContractError(f"{scenario_id}: snapshot belongs to another issue")
        _text(scenario.get("model_version"), f"{scenario_id}.model_version")
        _obj(scenario.get("assumptions"), f"{scenario_id}.assumptions")
        output = _obj(scenario.get("model_output"), f"{scenario_id}.model_output")
        if "fair_value" in output or "recommendation" in output:
            raise BenchmarkContractError(f"{scenario_id}: fair_value/recommendation assertions are forbidden")
        for field in ("bond_floor", "conversion_value", "option_component", "var", "expected_shortfall"):
            _number(output.get(field), f"{scenario_id}.model_output.{field}")

    total = len(issues)
    return {
        "schema_version": SCHEMA_VERSION,
        "issuer_count": len(issuers),
        "issue_count": total,
        "evidence_count": len(evidence),
        "field_evidence_complete_count": evidence_complete,
        "field_evidence_coverage": (evidence_complete / total) if total else 0.0,
        "scenario_ready_count": scenario_ready,
        "commercial_demo_ready": bool(total >= 10 and len(issuers) >= 5 and evidence_complete == total and scenario_ready >= 3),
    }

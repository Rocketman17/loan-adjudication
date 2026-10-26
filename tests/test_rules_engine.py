import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rules_engine import RuleEngine

RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "loan_rules.json"


def base_app() -> dict:
    """A minimally passing application."""
    return {
        "applicant": {
            "age": 30,
            "credit_score": 720,
            "annual_income_usd": 80000,
            "debt_to_income_ratio_percent": 25,
            "employment_status": "employed_full_time",
            "current_employment_duration_months": 24,
            "residency_status": "US_Citizen",
            "has_bankruptcy_recent": False,
            "has_verifiable_bank_account": True,
        },
        "loan_application": {"requested_amount_usd": 20000},
    }


def engine() -> RuleEngine:
    return RuleEngine(json.loads(RULES_PATH.read_text()))


# pass path
def test_all_pass_is_approve():
    e = engine()
    out = e.adjudicate(base_app())
    assert out["decision"] == "APPROVE"
    assert out["failed_rules"] == []
    assert len(out["evaluated"]) == 10


# every rule fails
def _with_field(app, path, value):
    app = copy.deepcopy(app)
    parts = path.split(".")
    cur = app
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return app


def test_age_below_18():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.age", 17))
    assert out["decision"] == "REJECT" and "RULE-AGE-001" in out["failed_rules"]


def test_credit_below_670():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.credit_score", 669))
    assert out["decision"] == "REJECT" and "RULE-CREDIT-001" in out["failed_rules"]


def test_credit_exactly_670_passes():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.credit_score", 670))
    assert out["decision"] == "APPROVE"


def test_income_below_30k():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.annual_income_usd", 29999.5))
    assert out["decision"] == "REJECT" and "RULE-INCOME-001" in out["failed_rules"]


def test_dti_above_40():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.debt_to_income_ratio_percent", 40.1))
    assert out["decision"] == "REJECT" and "RULE-DTI-001" in out["failed_rules"]


def test_dti_exactly_40_passes():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.debt_to_income_ratio_percent", 40))
    assert out["decision"] == "APPROVE"


def test_unemployed_major_fails_reject():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.employment_status", "unemployed"))
    assert out["decision"] == "REJECT" and "RULE-EMPLOY-001" in out["failed_rules"]


def test_tenure_below_6_flagged():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.current_employment_duration_months", 5))
    assert out["decision"] == "FLAG_REVIEW" and out["failed_rules"] == ["RULE-EMPLOY-002"]


def test_residency_fails():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.residency_status", "Visa_Holder"))
    assert out["decision"] == "REJECT" and "RULE-RESIDENCY-001" in out["failed_rules"]


def test_bankruptcy_true_fails():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.has_bankruptcy_recent", True))
    assert out["decision"] == "REJECT" and "RULE-BANKRUPTCY-001" in out["failed_rules"]


def test_loan_too_big_flagged():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "loan_application.requested_amount_usd", 40001))
    assert out["decision"] == "FLAG_REVIEW" and out["failed_rules"] == ["RULE-LOANAMT-001"]


def test_loan_exactly_half_income_passes():
    e = engine()
    app = _with_field(base_app(), "loan_application.requested_amount_usd", 40000)
    out = e.adjudicate(app)
    assert "RULE-LOANAMT-001" in [r["rule_id"] for r in out["evaluated"]
                                  if r["outcome"] == "PASS"]


def test_no_bank_account_fails():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.has_verifiable_bank_account", False))
    assert out["decision"] == "REJECT" and "RULE-BANKACCTS-001" in out["failed_rules"]


# aggregation
def test_reject_beats_flag_review():
    e = engine()
    app = _with_field(base_app(), "applicant.credit_score", 500)
    app = _with_field(app, "applicant.current_employment_duration_months", 1)
    out = e.adjudicate(app)
    assert out["decision"] == "REJECT"
    # FLAG_REVIEW rule failed but is not decision-binding
    assert "RULE-EMPLOY-002" not in out["failed_rules"]
    assert "RULE-EMPLOY-002" in out["failed_rules_all"]


def test_missing_field_counts_as_fail():
    e = engine()
    app = base_app()
    del app["applicant"]["age"]
    out = e.adjudicate(app)
    assert out["decision"] == "REJECT" and "RULE-AGE-001" in out["failed_rules"]


def test_zero_income_is_real_value_not_missing():
    e = engine()
    out = e.adjudicate(_with_field(base_app(), "applicant.annual_income_usd", 0))
    assert out["decision"] == "REJECT" and "RULE-INCOME-001" in out["failed_rules"]


def test_derived_threshold_recorded_for_audit():
    e = engine()
    out = e.adjudicate(base_app())
    rule = next(r for r in out["evaluated"] if r["rule_id"] == "RULE-LOANAMT-001")
    assert rule["threshold"] == 40000
    assert "0.5 x applicant.annual_income_usd" in rule["detail"]

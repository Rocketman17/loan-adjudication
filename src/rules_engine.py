"""Schema-driven deterministic rule engine for loan adjudication.

Design principles:
  * The ruleset JSON is the single source of truth - nothing about the
    individual rules is hard-coded; operators/fields/severities are read
    generically.
  * Every evaluation produces a full audit record (evaluated rule, actual
    value, threshold outcome) so the explanation layer never "invents" facts.
  * Decision policy:
        any failure with action_on_fail == REJECT      -> REJECT
        else any failure with action_on_fail FLAG_REVIEW -> FLAG_REVIEW
        else                                            -> APPROVE
  * Missing / null field -> the rule cannot be satisfied -> FAIL
    (data-completeness policy, documented in README).
"""

import copy
import operator
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- operators
_COMPARE_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}

_OP_HUMAN = {
    ">=": "at least", ">": "greater than", "<=": "at most",
    "<": "less than", "is": "equal to", "==": "equal to", "in": "one of",
}


def _dig(obj: Dict[str, Any], dotted_path: str) -> Any:
    """Resolve a dotted path like 'applicant.credit_score' into nested dicts."""
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _is_present(value: Any) -> bool:
    """A field is 'present' unless missing or explicitly null.
    Everything else (0, False, \"\") is a real value."""
    return value is not None


class RuleEngine:
    def __init__(self, ruleset: Dict[str, Any]):
        wrapper = ruleset.get("personal_loan_credit_rules", ruleset)
        self.version: str = str(wrapper.get("version", "unknown"))
        self.rules: List[Dict[str, Any]] = wrapper.get("rules", [])
        if not self.rules:
            raise ValueError("Ruleset contains no rules")
        # Map dotted path -> parsed numeric bounds, useful for synth data.
        self.field_paths: List[str] = [r["field"] for r in self.rules]

    # ------------------------------------------------------------ evaluation
    def _eval_one(self, rule: Dict[str, Any], app: Dict[str, Any]) -> Dict[str, Any]:
        rid = rule["id"]
        op = rule["operator"]
        actual = _dig(app, rule["field"])

        result = {
            "rule_id": rid,
            "rule_name": rule.get("name", rid),
            "group": rule.get("group"),
            "severity": rule.get("severity"),
            "action_on_fail": rule.get("action_on_fail"),
            "field": rule["field"],
            "actual_value": actual,
            "outcome": "PASS",
            "detail": "",
        }

        # ---- missing-data policy: unevaluable rule != satisfied rule
        if not _is_present(actual):
            result["outcome"] = "FAIL"
            result["detail"] = (f"'{rule['field']}' is missing in the application; "
                                f"the rule cannot be satisfied without it")
            return result

        if op in _COMPARE_OPS:
            if "value" in rule:
                threshold = {"value": rule["value"]}
                result["threshold"] = rule["value"]
            elif "value_field_multiplier" in rule:
                base = _dig(app, rule["value_field_multiplier"])
                if not _is_present(base):
                    result["outcome"] = "FAIL"
                    result["detail"] = (f"cannot compute the threshold because "
                                        f"'{rule['value_field_multiplier']}' is "
                                        f"missing in the application")
                    return result
                computed = float(base) * float(rule["multiplier_value"])
                threshold = {"value": round(computed, 2),
                             "derived_from": rule["value_field_multiplier"],
                             "multiplier": rule["multiplier_value"]}
                result["threshold"] = threshold["value"]
            else:
                raise ValueError(f"rule {rid}: no comparison value defined")

            passed = _COMPARE_OPS[op](actual, threshold["value"])
            result["outcome"] = "PASS" if passed else "FAIL"
            if "derived_from" in threshold:
                result["detail"] = (
                    f"{rule['field']} = {actual}; required to be "
                    f"{_OP_HUMAN[op]} {threshold['value']} "
                    f"({threshold['multiplier']} x "
                    f"{threshold['derived_from']} = {result['threshold']})")
            else:
                result["detail"] = (
                    f"{rule['field']} = {actual}; required to be "
                    f"{_OP_HUMAN[op]} {result['threshold']}")
            return result

        if op == "in":
            allowed = rule["value"]
            passed = actual in allowed
            result["outcome"] = "PASS" if passed else "FAIL"
            result["threshold"] = list(allowed)
            result["detail"] = (f"{rule['field']} = '{actual}'; "
                                f"allowed values: {', '.join(map(str, allowed))}")
            return result

        if op == "is":
            expected = rule["value"]          # explicit True/False contract
            ok_types = isinstance(expected, bool) and isinstance(actual, bool)
            passed = ok_types and (actual == expected)
            result["outcome"] = "PASS" if passed else "FAIL"
            result["threshold"] = expected
            result["detail"] = (f"{rule['field']} = {actual}; required value: "
                                f"{expected}")
            return result

        raise ValueError(f"rule {rid}: unsupported operator '{op}'")

    # ------------------------------------------------------------- public API
    def adjudicate(self, application: Dict[str, Any],
                   track_version: bool = True
                   ) -> Dict[str, Any]:
        """Full adjudication: decision + complete audit trail."""
        app = copy.deepcopy(application)

        evaluated = [self._eval_one(r, app) for r in self.rules]

        fails_reject = [e for e in evaluated
                        if e["outcome"] == "FAIL" and e["action_on_fail"] == "REJECT"]
        fails_flag = [e for e in evaluated
                      if e["outcome"] == "FAIL" and e["action_on_fail"] == "FLAG_REVIEW"]

        if fails_reject:
            decision, binding = "REJECT", fails_reject
        elif fails_flag:
            decision, binding = "FLAG_REVIEW", fails_flag
        else:
            decision, binding = "APPROVE", []

        return {
            "decision": decision,
            # rules whose failure BINDS the final decision (what the rationale
            # must cite). For REJECT, FLAG_REVIEW-only failures are moot.
            "failed_rules": [e["rule_id"] for e in binding],
            # every rule that returned FAIL (superset of failed_rules)
            "failed_rules_all": [e["rule_id"] for e in evaluated
                                 if e["outcome"] == "FAIL"],
            "evaluated": evaluated,
            "ruleset_version": self.version if track_version else None,
            "summary": _summary_line(decision, binding),
        }


def _summary_line(decision: str, binding: List[Dict[str, Any]]) -> str:
    if decision == "APPROVE":
        return "All credit rules satisfied."
    ids = [e["rule_id"] for e in binding]
    return f"Decision driven by: {', '.join(ids)}."

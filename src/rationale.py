"""Golden-rationale template builder.

Turns the engine's audit trail into a natural-language adjudication rationale,
deterministically. Serves three purposes:
  1. SFT *target* text (the fine-tuned LLM learns to reproduce this quality of
     grounded explanation).
  2. Faithfulness reference: any LLM-generated rationale is compared against
     the facts cited here.
  3. Deterministic fallback for the runtime path (fallback ladder in README).
"""


def _fmt_rule(e: dict) -> str:
    name = f"{e['rule_name']} ({e['rule_id']})"
    return f"{name}: {e['detail']}"


def build_rationale(adv: dict) -> str:
    decision = adv["decision"]
    evaluated = adv["evaluated"]
    failed = adv["failed_rules"]

    if decision == "APPROVE":
        parts = [f"{e['rule_name']}: {e['detail']}" for e in evaluated]
        return ("APPROVED - the application satisfies all "
                f"{len(evaluated)} credit rules. "
                + " ".join(parts)
                + " No rule was failed, so no manual review is required.")

    if decision == "REJECT":
        binding = [e for e in evaluated if e["rule_id"] in failed]
        parts = "; ".join(_fmt_rule(e) for e in binding)
        return (f"REJECTED - {len(binding)} critical or major rule(s) "
                f"failed. {parts}. "
                "A rejected application is not routed to manual review; "
                "failures of lower-severity rules are recorded for audit "
                "but do not drive this decision.")

    # FLAG_REVIEW
    binding = [e for e in evaluated if e["rule_id"] in failed]
    parts = "; ".join(_fmt_rule(e) for e in binding)
    return (f"FLAGGED FOR MANUAL REVIEW - all critical requirements were "
            f"met, but {len(binding)} lower-severity rule(s) "
            f"({', '.join(failed)}) failed. {parts}. "
            "An underwriter should review these items before a final "
            "approval is issued.")

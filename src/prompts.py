"""Prompt factory shared by all LLM paths (SFT training, zero/few-shot baseline
evaluations, and the runtime API).

Keeping ONE canonical prompt guarantees that:
  * the fine-tuned model is trained on the exact prompt format used in prod,
  * baseline-vs-SFT comparisons are apples-to-apples,
  * the audit chain (ruleset version) is embedded in every decision.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_ruleset() -> str:
    """Compact ruleset JSON string (embedded in the system prompt)."""
    rules = json.loads((ROOT / "data" / "loan_rules.json").read_text())
    return json.dumps(rules, separators=(",", ":"))


SYSTEM_TEMPLATE = """You are a credit adjudication assistant for a US personal-loan lender. You evaluate loan applications against the official ruleset provided below.

RULESET:
{ruleset}

TASK: Given an application JSON, output EXACTLY ONE JSON object with these keys:
- "decision": "APPROVE" | "FLAG_REVIEW" | "REJECT"
- "failed_rules": list of failed rule IDs whose failure drives the decision
- "rationale": 2-5 sentences citing each failed rule by its ID, the actual field value, and the required threshold. If APPROVE, briefly summarize that all rules passed.

DECISION POLICY: any REJECT-severity rule failing -> REJECT; else any FLAG_REVIEW rule failing -> FLAG_REVIEW; else APPROVE. A missing field must be treated as failing the rule that needs it. Do not output anything outside the JSON object."""

FEW_SHOT_HEADER = """Here are worked examples:

"""

ASSISTANT_PROTOTYPE = """{"decision": "{decision}", "failed_rules": {failed}, "rationale": "{rationale}"}"""


def app_to_user_message(app_json: str) -> str:
    return f"APPLICATION:\n{app_json}\n\nAdjudicate the application above and respond with the JSON object."


def build_messages(application_str: str, examples=None) -> list:
    """examples: list of (application_json_str, assistant_output_str)."""
    msgs = [{"role": "system", "content": SYSTEM_TEMPLATE.format(ruleset=load_ruleset())}]
    if examples:
        shots = []
        for app_s, out_s in examples:
            shots.append(json.dumps({"application": json.loads(app_s)}))
            shots.append(out_s)
        msgs[0]["content"] += "\n" + FEW_SHOT_HEADER + "\n\n".join(shots)
    msgs.append({"role": "user", "content": app_to_user_message(application_str)})
    return msgs

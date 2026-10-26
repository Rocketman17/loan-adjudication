"""Step 1 of SFT data prep: build instruction-response pairs.

Input : data/synthetic/{train,val,test}.jsonl (engine-labelled)
Output: data/sft/{train,val}.jsonl in OpenAI-style "messages" format -
        the template-agnostic intermediate that any tokenizer chat-template
        can render (tokenization itself happens inside train_sft.py, which
        reports padding/truncation statistics).

The assistant target is the ground-truth JSON emitted by the rules engine
and the golden rationale template - the model learns to *quote* engine-grade
facts instead of inventing them.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import SYSTEM_TEMPLATE, app_to_user_message  # noqa: E402
from rationale import build_rationale  # noqa: E402
from rules_engine import RuleEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def build_target(label_decision: str, failed_rules: list,
                 engine_audit: dict) -> str:
    return json.dumps({
        "decision": label_decision,
        "failed_rules": failed_rules,
        "rationale": build_rationale(engine_audit),
    }, separators=(",", ": "), ensure_ascii=False)


def main():
    engine = RuleEngine(json.loads(
        (ROOT / "data" / "loan_rules.json").read_text()))

    (ROOT / "data" / "sft").mkdir(exist_ok=True)
    for split in ["train", "val"]:
        recs = [json.loads(l) for l in
                (ROOT / "data" / "synthetic" / f"{split}.jsonl").read_text().splitlines() if l]
        out_lines = []
        for r in recs:
            app_json = json.dumps(r["input"])
            audit = engine.adjudicate(r["input"])
            # sanity: synthesis label vs live engine (guards against engine edits)
            assert audit["decision"] == r["label"]["decision"], "engine drift!"
            target = build_target(audit["decision"], audit["failed_rules"], audit)
            out_lines.append(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM_TEMPLATE.format(
                        ruleset=json.dumps(json.loads(
                            (ROOT / "data" / "loan_rules.json").read_text()),
                            separators=(",", ":")))},
                    {"role": "user", "content": app_to_user_message(app_json)},
                    {"role": "assistant", "content": target},
                ]
            }, ensure_ascii=False))
        path = ROOT / "data" / "sft" / f"{split}.jsonl"
        path.write_text("\n".join(out_lines) + "\n")
        print(f"{split}: wrote {len(out_lines)} pairs -> {path}")

    # test split is kept as engine-labelled raw for evaluation
    print("test split left as-is for evaluation (not part of training).")


if __name__ == "__main__":
    main()

"""Evaluation harness.

Measures, for any adjudicator (engine / zero-shot / few-shot / SFT-LoRA):

 1. decision metrics         accuracy + per-class P/R/F1 over 3 decisions
 2. per-rule identification  macro-F1 of each rule id being cited exactly when
                             the engine says that rule binds the decision
 3. boundary slices          accuracy within +-band of each numeric threshold
                             (the failure surface of probabilistic models)
 4. rationale faithfulness   precision/recall of rule-id citations inside free
                             text, checked against engine truth

Writes eval/{name}_predictions.jsonl and merges metrics into eval/metrics.json.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infer import DECISIONS  # noqa: E402
from rules_engine import RuleEngine  # noqa: E402

RULE_ID_RE = re.compile(r"RULE-[A-Z]+-\d+")


def load_test():
    return [json.loads(l) for l in
            (ROOT / "data" / "synthetic" / "test.jsonl").read_text().splitlines() if l]


# -------------------------------------------------------------- metrics core
def prf(cm: Counter, cls: str) -> dict:
    tp = cm[(cls, cls)]
    fp = sum(v for (p, t), v in cm.items() if p == cls and t != cls)
    fn = sum(v for (p, t), v in cm.items() if p != cls and t == cls)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "support": tp + fn}


def decision_metrics(pairs):
    """pairs: list of (pred: dict|None, gold: dict) - pred None = parse fail."""
    cm = Counter()
    n = len(pairs)
    n_undec = 0
    for pred, gold in pairs:
        p = pred["decision"] if pred and pred["decision"] else "UNPARSED"
        if p == "UNPARSED":
            n_undec += 1
        cm[(p, gold["label"]["decision"])] += 1
    n_correct = sum(v for (p, t), v in cm.items() if p == t)
    acc = n_correct / n if n else 0.0
    macro = {}
    # treat UNPARSED as its own class (hurts the offending class)
    per = {c: prf(cm, c) for c in sorted(set(DECISIONS) | {"UNPARSED"})
           if cm.get((c, c), 0) or any(t == c or p == c for (p, t) in cm)}
    macro_f1 = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "n": n, "unparsed": n_undec, "per_class": per,
            "confusion": {f"{p}<-{t}": v for (p, t), v in cm.items() if v}}


def rule_identification_metrics(pairs):
    """Macro-F1 over the 10 rules: is rule X cited exactly when engine says
    it binds (in gold.failed_rules)? UNPARSED predictions score 0."""
    per_rule = {}
    all_ids = sorted({rid for _, g in pairs for rid in g["label"]["failed_rules"]} |
                     {rid for p, _ in pairs for rid in (p or {}).get("failed_rules", [])})
    for rid in all_ids:
        tp = fp = fn = 0
        for pred, gold in pairs:
            pred_cited = set((pred or {}).get("failed_rules", []))
            gold_bind = set(gold["label"]["failed_rules"])
            if rid in pred_cited and rid in gold_bind:
                tp += 1
            elif rid in pred_cited:
                fp += 1
            elif rid in gold_bind:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_rule[rid] = {"precision": round(prec, 4), "recall": round(rec, 4),
                         "f1": round(f1, 4), "support": tp + fn}
    macro = sum(v["f1"] for v in per_rule.values()) / len(per_rule)
    return {"macro_f1_rules": round(macro, 4), "per_rule": per_rule}


def boundary_slices(pairs, engine: RuleEngine):
    """Accuracy split by proximity to a numeric threshold. Only applicable to
    numeric rules with a fixed threshold; the derived loan rule uses
    0.5x income. Engine is re-consulted per record (cheap)."""
    numeric = []
    for rule in engine.rules:
        if rule["operator"] in (">=", "<="):
            numeric.append(rule)
    out = {}
    for rule in numeric:
        rid, field = rule["id"], rule["field"]
        near_correct = far_correct = 0
        near = far = 0
        for pred, gold in pairs:
            node = gold["input"]
            try:
                for part in field.split("."):
                    node = node[part]
                actual = float(node)
            except (KeyError, TypeError, ValueError):
                continue
            if "value_field_multiplier" in rule:
                base = gold["input"].get("applicant", {}).get("annual_income_usd")
                if base is None:
                    continue
                threshold = float(base) * rule["multiplier_value"]
                band = 0.03 * threshold
            else:
                threshold = float(rule["value"])
                band = max(1.0, 0.02 * abs(threshold))
            is_near = abs(actual - threshold) <= band
            correct = (pred or {}).get("decision") == gold["label"]["decision"]
            if is_near:
                near += 1
                near_correct += correct
            else:
                far += 1
                far_correct += correct
        if near + far:
            out[rid] = {
                "near": {"n": near, "acc": round(near_correct / near, 3) if near else 0},
                "far": {"n": far, "acc": round(far_correct / far, 3) if far else 0},
                "threshold": rule.get("value", "0.5x income"),
                "band": "±2-3%",
            }
    return out


def rationale_faithfulness(pairs):
    """Do rule citations in free text match engine truth?
    precision: share of cited ids that failed at all in the engine
    recall:    share of BINDING (decision-driving) ids that were cited"""
    precs, recs, covers = [], [], []
    for pred, gold in pairs:
        text = (pred or {}).get("rationale", "") or ""
        cited = set(RULE_ID_RE.findall(text))
        gold_all = set(gold["label"]["failed_rules_all"])
        gold_bind = set(gold["label"]["failed_rules"])
        if not cited:
            if gold_bind:
                covers.append(0.0)
            continue
        wrong = cited - gold_all
        precs.append(1.0 - len(wrong) / len(cited))
        recs.append(len(cited & gold_bind) / len(gold_bind) if gold_bind else 0.0)
        covers.append(1.0 if (gold_bind <= cited) else 0.0)
    n = len(pairs) or 1
    return {
        "n": n,
        "citation_precision": round(sum(precs) / max(len(precs), 1), 4),
        "binding_recall": round(sum(recs) / max(len(recs), 1), 4),
        "fully_faithful_share": round(sum(covers) / n, 4),
    }


def full_report(name, pairs, engine) -> dict:
    report = {"name": name}
    report.update(decision_metrics(pairs))
    report.update(rule_identification_metrics(pairs))
    report["boundary_slices"] = boundary_slices(pairs, engine)
    report["faithfulness"] = rationale_faithfulness(pairs)
    _save(name, pairs, report)
    return report


def _save(name, pairs, report):
    (ROOT / "eval").mkdir(exist_ok=True)
    with (ROOT / "eval" / f"{name}_predictions.jsonl").open("w") as f:
        for pred, gold in pairs:
            f.write(json.dumps({
                "pred": pred, "gold_label": gold["label"]
            }, ensure_ascii=False) + "\n")
    _merge_metrics(name, report)


def _merge_metrics(name, report):
    path = ROOT / "eval" / "metrics.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = report
    path.write_text(json.dumps(data, indent=2))

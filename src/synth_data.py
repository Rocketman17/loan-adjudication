"""Synthetic personal-loan application generator.

Strategy:
  * realistic marginal distributions per field,
  * mild correlations (income <-> DTI, income <-> loan amount),
  * stratified boundary injection around EVERY threshold (this is where a
    probabilistic model is most likely to break, and exactly what we audit),
  * deliberate share of missing-field and multi-failure records,
  * decision quotas enforced by accept/resample against the engine.
"""

import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules_engine import RuleEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42

EMPLOYMENT_STATUSES = [
    ("employed_full_time", 0.62), ("self_employed", 0.12),
    ("employed_part_time", 0.10), ("retired", 0.08),
    ("unemployed", 0.05), ("student", 0.03),
]
RESIDENCY_STATUSES = [
    ("US_Citizen", 0.82), ("Permanent_Resident", 0.10),
    ("Visa_Holder", 0.05), ("Other", 0.03),
]


def _rev_dist(dist, rng):
    r, acc = rng.random(), 0.0
    for v, p in dist:
        acc += p
        if r <= acc:
            return v
    return dist[-1][0]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_candidate(rng: random.Random) -> dict:
    """One 'demographic personality': internally correlated fields."""
    income = round(_clamp(rng.gauss(62000, 32000), 8000, 250000), -2)
    # DTI mildly anti-correlated with income (poor -> more burdened)
    dti_base = rng.gauss(38 - 10 * min(income / 80000, 1.0), 10)
    dti = round(_clamp(dti_base, 2, 78), 1)
    loan_ratio = _clamp(rng.gauss(0.28, 0.17), 0.02, 0.72)
    requested = round(income * loan_ratio, -2)

    credit = int(_clamp(
        rng.gauss(665, 75) - (400 if rng.random() < 0.06 else 0),  # bankruptcy-ish
        300, 850))
    if rng.random() < 0.30:                 # fresh uniform chunk, breaks clumping
        credit = rng.randint(300, 850)

    age = rng.choices(
        [rng.randint(16, 17), rng.randint(18, 28), rng.randint(29, 55),
         rng.randint(56, 90)],
        weights=[0.03, 0.30, 0.47, 0.20])[0]

    # young / unemployed / low-scorers are likelier to have short tenure
    tenure = int(_clamp(rng.expovariate(1 / 40), 0, 480))
    if rng.random() < 0.12:
        tenure = rng.randint(0, 8)

    app = {
        "applicant": {
            "age": age,
            "credit_score": credit,
            "annual_income_usd": income,
            "debt_to_income_ratio_percent": dti,
            "employment_status": _rev_dist(EMPLOYMENT_STATUSES, rng),
            "current_employment_duration_months": tenure,
            "residency_status": _rev_dist(RESIDENCY_STATUSES, rng),
            "has_bankruptcy_recent": rng.random() < 0.06,
            "has_verifiable_bank_account": rng.random() > 0.05,
        },
        "loan_application": {"requested_amount_usd": int(requested)},
    }
    return app


# ------------------------------------------------------------ boundary bands
# (field, band around the threshold, engine-observable weight multiplier)
BOUNDARY_SPECS = {
    "RULE-AGE-001":          ("applicant.age", 1.0, 3.0),
    "RULE-CREDIT-001":       ("applicant.credit_score", 5.0, 3.0),
    "RULE-INCOME-001":       ("applicant.annual_income_usd", 2000.0, 3.0),
    "RULE-DTI-001":          ("applicant.debt_to_income_ratio_percent", 2.5, 3.0),
    "RULE-EMPLOY-002":       ("applicant.current_employment_duration_months", 1.0, 3.0),
    "RULE-LOANAMT-001":      ("loan_application.requested_amount_usd", None, 3.0),
}


def _make_boundary_variant(app: dict, rule_id: str, rng) -> dict:
    """Nudge one field into the +-band of its threshold (both sides)."""
    app = copy.deepcopy(app)
    field, band, _ = BOUNDARY_SPECS[rule_id]
    parts = field.split(".")
    cur = app
    for p in parts[:-1]:
        cur = cur[p]
    half = rng.choice([-1, 1])
    if rule_id == "RULE-AGE-001":
        cur[parts[-1]] = 18 + half * rng.choice([0, 0, 1])
    elif rule_id == "RULE-CREDIT-001":
        cur[parts[-1]] = _clamp(670 + half * rng.choice([1, 3, 5]), 300, 850)
    elif rule_id == "RULE-INCOME-001":
        base = app["applicant"]["annual_income_usd"]
        cur[parts[-1]] = _clamp(30000 + half * rng.choice([250, 1000, 2000]),
                                8000, 250000)
        # keep the loan amount consistent with the (possibly changed) income
        app["loan_application"]["requested_amount_usd"] = int(
            round(app["applicant"]["annual_income_usd"] *
                  _clamp(rng.gauss(0.28, 0.15), 0.03, 0.6), -2))
    elif rule_id == "RULE-DTI-001":
        cur[parts[-1]] = _clamp(40 + half * rng.choice([0.5, 1.5, 2.5]), 2, 78)
    elif rule_id == "RULE-EMPLOY-002":
        cur[parts[-1]] = _clamp(6 + half * rng.choice([0, 1, 2]), 0, 480)
    elif rule_id == "RULE-LOANAMT-001":
        income = float(app["applicant"]["annual_income_usd"])
        cur[parts[-1]] = int(round(
            income * (0.5 + half * rng.choice([0.01, 0.03, 0.06])), -2))
    return app


SPECIAL_FIELDS = [
    "applicant.age", "applicant.credit_score", "applicant.annual_income_usd",
    "applicant.debt_to_income_ratio_percent", "applicant.employment_status",
    "applicant.current_employment_duration_months",
    "applicant.residency_status", "applicant.has_bankruptcy_recent",
    "applicant.has_verifiable_bank_account",
    "loan_application.requested_amount_usd",
]


def _drop_field(app: dict, field: str) -> dict:
    app = copy.deepcopy(app)
    parts = field.split(".")
    cur = app
    for p in parts[:-1]:
        cur = cur[p]
    cur.pop(parts[-1], None)
    return app


def generate(N: int, engine: RuleEngine, rng: random.Random,
             quota: dict = None) -> list:
    """Generate N engine-labelled records with decision quotas and boundary
    stratification."""
    quota = quota or {"APPROVE": 0.30, "FLAG_REVIEW": 0.18, "REJECT": 0.52}
    buckets = {"APPROVE": [], "FLAG_REVIEW": [], "REJECT": []}
    target = {k: int(N * v) for k, v in quota.items()}

    while min(len(buckets[k]) for k in buckets) < 1 or \
            any(len(buckets[k]) < target[k] for k in buckets):
        app = make_candidate(rng)

        # 10% of the time, push a random boundary rule into its band
        if rng.random() < 0.35:
            rid = rng.choice(list(BOUNDARY_SPECS))
            app = _make_boundary_variant(app, rid, rng)

        adv = engine.adjudicate(app)
        label = adv["decision"]
        if len(buckets[label]) < target[label]:
            rec = {"input": app,
                   "label": {"decision": label,
                             "failed_rules": adv["failed_rules"],
                             "failed_rules_all": adv["failed_rules_all"]}}
            buckets[label].append(rec)

    out = []
    for k in buckets:
        out.extend(buckets[k])

    # 3% missing-field records (data-completeness edge cases)
    n_missing = int(N * 0.03)
    existing_missing = sum(1 for r in out
                           if any(f not in json.dumps(r["input"]) for f in []))
    for _ in range(max(0, n_missing - existing_missing)):
        app = _drop_field(make_candidate(rng), rng.choice(SPECIAL_FIELDS))
        adv = engine.adjudicate(app)
        out.append({"input": app,
                    "label": {"decision": adv["decision"],
                              "failed_rules": adv["failed_rules"],
                              "failed_rules_all": adv["failed_rules_all"]}})
    rng.shuffle(out)
    return out[:N]


def main():
    rng = random.Random(RANDOM_SEED)
    engine = RuleEngine(json.loads(
        (ROOT / "data" / "loan_rules.json").read_text()))

    sets = {"train": 3000, "val": 500, "test": 500}
    stats = {}
    for name, n in sets.items():
        records = generate(n, engine, rng)
        path = ROOT / "data" / "synthetic" / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        counts = {}
        for r in records:
            counts[r["label"]["decision"]] = \
                counts.get(r["label"]["decision"], 0) + 1
        stats[name] = counts
        print(f"{name}: {len(records)} records -> {counts}")

    # ---- sanity: a human-readable spot check of 3 examples
    (ROOT / "eval" / "synth_spotcheck.json").write_text(json.dumps(
        {"distribution": stats,
         "note": "labels generated exclusively by rules_engine; see README"},
        indent=2))
    print("wrote data/synthetic/{train,val,test}.jsonl")


if __name__ == "__main__":
    main()

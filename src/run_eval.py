"""Run baseline adjudicator EVAL on a subset of the test split.

Usage:
  python3 src/run_eval.py --mode engine
  python3 src/run_eval.py --mode zero_shot   [--n 200]
  python3 src/run_eval.py --mode few_shot    [--n 200]
  python3 src/run_eval.py --mode sft         --adapter models/lora-sft/adapter
    (sft runs the LoRA adapter with transformers inside the training image)
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import full_report, load_test  # noqa: E402
from infer import OllamaClient, TransformersClient, adjudicate_llm, few_shot_examples  # noqa: E402
from prompts import build_messages  # noqa: E402
from rules_engine import RuleEngine  # noqa: E402


def engine_mode(test, engine):
    pairs = []
    for r in test:
        audit = engine.adjudicate(r["input"])
        pred = {"decision": audit["decision"],
                "failed_rules": audit["failed_rules"],
                "rationale": audit["summary"], "parse_ok": True}
        pairs.append((pred, r))
    return full_report("engine", pairs, engine)


def llm_mode(test, engine, client, name, n):
    recs = test[:n]
    pairs = []
    t0 = time.time()
    for i, r in enumerate(recs):
        app_str = json.dumps(r["input"])
        msgs = build_messages(app_str, examples=None)
        try:
            pred = adjudicate_llm(client, msgs)
        except Exception as e:  # network / model errors -> unparsed
            pred = {"decision": None, "failed_rules": [], "rationale": "",
                    "parse_ok": False}
        pairs.append((pred, r))
        if (i + 1) % 25 == 0:
            print(f"  {name}: {i + 1}/{len(recs)} done, "
                  f"{(time.time() - t0) / (i + 1):.1f}s/req", flush=True)
    return full_report(name, pairs, engine)


def llm_mode_few(test, engine, client, name, n):
    examples = few_shot_examples(engine)
    recs = test[:n]
    pairs = []
    for i, r in enumerate(recs):
        msgs = build_messages(json.dumps(r["input"]), examples=examples)
        try:
            pred = adjudicate_llm(client, msgs)
        except Exception:
            pred = {"decision": None, "failed_rules": [], "rationale": "",
                    "parse_ok": False}
        pairs.append((pred, r))
        if (i + 1) % 25 == 0:
            print(f"  {name}: {i + 1}/{len(recs)}", flush=True)
    return full_report(name, pairs, engine)


def sft_mode(test, engine, adapter_path, n):
    client = TransformersClient(model_id="Qwen/Qwen2.5-1.5B-Instruct", adapter_path=adapter_path)
    recs = test[:n]
    pairs = []
    t0 = time.time()
    for i, r in enumerate(recs):
        app_str = json.dumps(r["input"])
        msgs = build_messages(app_str, examples=None)
        try:
            pred = adjudicate_llm(client, msgs)
        except Exception as e:
            pred = {"decision": None, "failed_rules": [], "rationale": "",
                    "parse_ok": False}
        pairs.append((pred, r))
        if (i + 1) % 25 == 0:
            print(f"  sft: {i + 1}/{len(recs)} done, "
                  f"{(time.time() - t0) / (i + 1):.1f}s/req", flush=True)
    return full_report("sft", pairs, engine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["engine", "zero_shot", "few_shot", "sft"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="qwen2.5:3b-instruct-q4_K_M")
    ap.add_argument("--adapter", help="Path to LoRA adapter")
    args = ap.parse_args()

    engine = RuleEngine(json.loads(
        (ROOT / "data" / "loan_rules.json").read_text()))
    test = load_test()

    if args.mode == "engine":
        rep = engine_mode(test, engine)
    elif args.mode == "zero_shot":
        rep = llm_mode(test, engine, OllamaClient(model=args.model),
                       "zero_shot", args.n)
    elif args.mode == "few_shot":
        rep = llm_mode_few(test, engine, OllamaClient(model=args.model),
                           "few_shot", args.n)
    elif args.mode == "sft":
        if not args.adapter:
            print("Error: --adapter is required for sft mode")
            sys.exit(1)
        rep = sft_mode(test, engine, args.adapter, args.n)
    
    print(json.dumps(rep, indent=2)[:1600])


if __name__ == "__main__":
    main()

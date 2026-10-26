"""LLM inference client.

The runtime decision path is ALWAYS the deterministic engine; the LLM is the
explainer + a baseline adjudicator whose quality we measure. This module:

  * talks to a local Ollama server (no data leaves the device),
  * enforces JSON-schema-ish parsing with a safe fallback,
  * exposes `adjudicate_llm(app)` returning the parsed triple or an error.

Engine stitching happens in evaluate.py / api.py:
  decision   <- rules_engine (canonical)
  failed     <- rules_engine
  rationale  <- LLM (faithfulness scored later by evaluate.py)
"""

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DECISIONS = {"APPROVE", "FLAG_REVIEW", "REJECT"}


class OllamaClient:
    def __init__(self, model="qwen2.5:3b-instruct-q4_K_M", host=DEFAULT_HOST,
                 temperature=0.0, max_tokens=256, seed=42):
        self.model = model
        self.app_host = host
        self.url = f"{host}/api/chat"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def chat(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "seed": self.seed,
            },
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())["message"]["content"]

class TransformersClient:
    """Local inference client using Transformers + PEFT."""
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", adapter_path: str = None,
                 temperature=0.0, max_tokens=512):
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from peft import PeftModel
        import torch
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        gc = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        self.model.generation_config = gc
        self.model.eval()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[dict]) -> str:
        # Convert chat messages to prompt format
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')

        with self.torch.no_grad():
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": self.max_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            outputs = self.model.generate(**gen_kwargs)

        # Extract only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return generated_text


def _extract_json(text: str) -> Optional[dict]:
    """First balanced JSON object in the text (robust to chatty wrappers)."""
    text = text.strip()
    m = re.search(r"\{", text)
    if not m:
        return None
    start = m.start()
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_prediction(raw: str) -> Dict[str, Any]:
    """Parse a raw model answer into a validated record.

    Returns dict with keys: decision, failed_rules, rationale, parse_ok.
    Never raises - invalid outputs degrade gracefully.
    """
    obj = _extract_json(raw)
    if not obj:
        return {"decision": None, "failed_rules": [],
                "rationale": "", "parse_ok": False}
    decision = obj.get("decision")
    if decision not in DECISIONS:
        decision = None
    fr = obj.get("failed_rules")
    if not isinstance(fr, list):
        fr = []
    fr = [r for r in fr if isinstance(r, str) and re.match(r"RULE-[A-Z]+-\d+$", r)]
    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return {"decision": decision, "failed_rules": fr, "rationale": rationale,
            "parse_ok": True}


def adjudicate_llm(client: Any, messages: List[dict]) -> dict:
    raw = client.chat(messages)
    out = parse_prediction(raw)
    out["raw"] = raw
    return out


def few_shot_examples(engine, path="data/synthetic/train.jsonl", k=3) -> list:
    """Deterministic few-shot exemplars, one per decision class.

    `engine` re-evalates each pick so the exemplar rationale quotes live
    engine facts (single source of truth even for shots).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rationale import build_rationale

    recs = [json.loads(l) for l in open(path) if l]
    picks, wanted = [], ["APPROVE", "REJECT", "FLAG_REVIEW"]
    for w in wanted:
        for r in recs:
            if r["label"]["decision"] == w and \
                    all(_recall_key(p[0]) != _recall_key(
                    json.dumps(r["input"])) for p in picks):
                audit = engine.adjudicate(r["input"])
                golden = json.dumps(
                    {"decision": w, "failed_rules": audit["failed_rules"],
                     "rationale": build_rationale(audit)}, ensure_ascii=False)
                picks.append((json.dumps(r["input"]), golden))
                break
    return picks[:k]


def _recall_key(a: str) -> str:
    return a

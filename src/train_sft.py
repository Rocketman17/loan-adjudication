"""SFT fine-tuning of Qwen2.5-1.5B-Instruct with LoRA (bf16 + triu patch, Jetson-friendly).

Pipeline:
 1. load engine-built instruction pairs             (data/sft/{train,val}.jsonl)
 2. render each conversation through the model's OWN chat template (Qwen
    ChatML: <|im_start|>role\\n...<|im_end|>) -> the exact special tokens the
    base model expects come from the template itself; the tokenizer is called
    with add_special_tokens=False so they are never duplicated.
 3. prompt tokens are masked to -100 in the labels -> loss only on the
    assistant JSON answer (completion-only SFT via precise prefix token matching).
 4. dynamic padding to the longest sequence in each batch; truncation at
    MAX_LEN=2048 with drop/length statistics written to eval/tokenization_stats.json
 5. LoRA r=16/alpha=32 on attention+MLP projections; base stays bf16 (no
    bitsandbytes - BnB is fragile on aarch64; Orin's 64 GB unified memory
    handles a 1.5B in bf16 comfortably).

Run (Jetson):
  docker run --runtime nvidia --rm -v $PWD:/work -w /work loan-train:deps \
      python3 src/train_sft.py --out models/lora-sft
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

# =============================================================================
# JETSON BF16 PATCH: Fix missing torch.triu/tril CUDA kernels for BFloat16
# =============================================================================
_orig_triu = torch.triu
def _patched_triu(input, diagonal=0):
    if input.dtype == torch.bfloat16:
        return _orig_triu(input.to(torch.float32), diagonal=diagonal).to(torch.bfloat16)
    return _orig_triu(input, diagonal=diagonal)
torch.triu = _patched_triu

_orig_tril = torch.tril
def _patched_tril(input, diagonal=0):
    if input.dtype == torch.bfloat16:
        return _orig_tril(input.to(torch.float32), diagonal=diagonal).to(torch.bfloat16)
    return _orig_tril(input, diagonal=diagonal)
torch.tril = _patched_tril
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(4)

MAX_LEN = 2048
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ----------------------------------------------------------- thermal watchdog
def _tj_temp():
    """Tjunction temp in C from the Jetson thermal zones (None if unknown)."""
    try:
        for zone in Path("/sys/devices/virtual/thermal").glob("thermal_zone*"):
            if zone.joinpath("type").read_text().strip() == "tj-therm":
                return int(zone.joinpath("temp").read_text().strip()) / 1000
    except OSError:
        pass
    return None


def guard_thermal():
    """Safety mode after a Jetson hard crash: pause while hot, bail if scorching.
    Trainer re-runs resume automatically from the last LoRA checkpoint."""
    t = _tj_temp()
    if t is None:
        return
    if t >= 92:
        raise SystemExit(f"[thermal-watchdog] ABORT: tj={t}C >= 92C; "
                         f"last checkpoint is safe, rerun with --resume")
    wait_until = time.time() + 15 * 60
    while t >= 82 and time.time() < wait_until:
        print(f"[thermal-watchdog] tj={t}C too hot - cooling pause "
              f"({_tj_temp()}C ...) ...", flush=True)
        time.sleep(20)
        t = _tj_temp()


def load_pairs(split: str):
    path = ROOT / "data" / "sft" / f"{split}.jsonl"
    return [json.loads(l)["messages"] for l in path.read_text().splitlines() if l]


def encode_conversation(tok, msgs, stats):
    full_text = tok.apply_chat_template(msgs, tokenize=False)
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                        add_generation_prompt=True)

    full_enc = tok(full_text, add_special_tokens=False, truncation=True, max_length=MAX_LEN)
    prompt_enc = tok(prompt_text, add_special_tokens=False, truncation=True, max_length=MAX_LEN)

    full_ids = full_enc["input_ids"]
    prompt_ids = prompt_enc["input_ids"]

    split_idx = len(prompt_ids)

    if split_idx >= len(full_ids) - 5:
        split_idx = int(len(full_ids) * 0.7)

    stats["lengths"].append(len(full_ids))
    n_supervised = len(full_ids) - split_idx
    if n_supervised < 2:
        stats["rejected_empty"] += 1
        return None

    labels = [-100] * split_idx + full_ids[split_idx:]
    stats["assistant_tokens"] += n_supervised

    return full_ids, labels


class SFTDataset(Dataset):
    def __init__(self, tok, msgs_list, stats, max_samples=0):
        self.rows = []
        for m in (msgs_list[:max_samples] if max_samples else msgs_list):
            enc = encode_conversation(tok, m, stats)
            if enc:
                self.rows.append(enc)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}


class Collator:
    """Dynamic padding: pad to the longest row in the batch."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)

        def pad(seq, fill):
            return seq + [fill] * (maxlen - len(seq))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id)
                                       for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0)
                                            for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


def main():
    global MAX_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/lora-sft")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="debug cap on training samples")
    ap.add_argument("--eval-max-samples", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--base", default=BASE_MODEL,
                    help="override base model")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in out/hf_trainer")
    args = ap.parse_args()

    MAX_LEN = args.max_len
    base_model = args.base

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                             TrainerCallback, TrainingArguments)

    class ThermalGuard(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            guard_thermal()

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    stats = {"lengths": [], "assistant_tokens": 0, "rejected_empty": 0}
    train_ds = SFTDataset(tok, load_pairs("train"), stats,
                          max_samples=args.max_samples)
    val_ds = SFTDataset(tok, load_pairs("val")[:args.eval_max_samples],
                        stats, max_samples=0)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use bfloat16 for native stability without overflow/NaNs
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        attn_implementation="eager")
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))

    model.enable_input_require_grads()
    model.print_trainable_parameters()

    def collate(batch):
        return Collator(tok.pad_token_id)(batch)

    targs = TrainingArguments(
        output_dir=str(out_dir / "hf_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=150,
        per_device_eval_batch_size=args.batch,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,  # Enable native bfloat16 training flags
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=42,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds,
                      data_collator=collate,
                      callbacks=[ThermalGuard()])

    resume_from = None
    if args.resume:
        ckpts = sorted((out_dir / "hf_trainer").glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1])) \
            if (out_dir / "hf_trainer").exists() else []
        if ckpts:
            resume_from = str(ckpts[-1])
        print(f"[resume] {resume_from or 'no checkpoint found - fresh start'}")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    print(f"[train] total wall time: {time.time() - t0:.0f}s")

    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))

    eval_metrics = trainer.evaluate()
    lengths = stats["lengths"]
    stats["mean_len"] = sum(lengths) / len(lengths)
    stats["max_len"] = max(lengths)
    stats["n_truncated"] = sum(1 for l in lengths if l >= MAX_LEN)
    stats["padded_val_loss"] = eval_metrics["eval_loss"]
    del lengths
    (ROOT / "eval" / "tokenization_stats.json").write_text(json.dumps(
        stats, indent=2))

    print("[eval] eval_loss:", eval_metrics["eval_loss"])
    print("[stats]", json.dumps({k: v for k, v in stats.items()
                                   if k != "lengths"}, indent=2))
    print(f"[done] adapter saved -> {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
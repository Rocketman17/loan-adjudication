# Loan Adjudication System

A fine-tuned LLM that evaluates personal loan applications and produces an adjudication decision (`APPROVE` / `FLAG_REVIEW` / `REJECT`) with a natural-language rationale explaining which rules passed or failed.

**Core idea:** A deterministic rules engine generates high-quality labeled training data. The fine-tuned LoRA adapter (Qwen2.5-1.5B-Instruct) learns to replicate this decision logic and explain its reasoning - acting as a standalone LLM adjudicator at inference time.

## Project Structure

```
loan-adjudication/
├── src/
│   ├── rules_engine.py     # Generates training labels (not used at inference)
│   ├── rationale.py        # Builds training rationales from engine output
│   ├── synth_data.py       # Generates synthetic loan applications
│   ├── sft_dataset.py      # Formats data for LLM fine-tuning
│   ├── train_sft.py        # LoRA fine-tuning (Qwen2.5-1.5B + PEFT)
│   ├── infer.py            # Runs the fine-tuned model
│   ├── evaluate.py         # Metrics: accuracy, F1, rule-level
│   ├── run_eval.py         # CLI eval runner
│   └── prompts.py          # Chat templates
├── data/
│   ├── loan_rules.json     # 10-rule ruleset
│   ├── synthetic/          # Generated applications (train/val/test)
│   └── sft/                # SFT pairs in chat format
├── tests/
│   └── test_rules_engine.py
├── eval/
│   └── metrics.json        # Pre-computed evaluation results
├── configs/
│   └── requirements-training.txt
├── README.md
└── REPORT.md
```

## Evaluation Results (n=200 test cases)

| Mode | Accuracy | Macro-F1 | Unparsed | Rule-F1 |
|---|---|---|---|---|
| **SFT - LoRA Qwen2.5-1.5B** | **0.725** | **0.545** | **0** | **0.907** |
| Zero-shot - qwen2.5:3b | 0.505 | 0.383 | 4 | 0.434 |
| Few-shot - qwen2.5:3b | 0.400 | 0.254 | 51 | 0.497 |
| Engine (ground truth) | 1.000 | 1.000 | 0 | 1.000 |

See `REPORT.md` for full analysis and per-class breakdown.

---

## Reproducing Results

### Prerequisites

- Python 3.10+
- CUDA GPU (for training and SFT inference)
- [Ollama](https://ollama.ai) installed (for zero/few-shot baselines only)

### 1. Install dependencies

```bash
pip install -r configs/requirements-training.txt
```

### 2. Run tests

```bash
python3 -m pytest tests/ -v
# Expected: 18 passed
```

### 3. Download the LoRA adapter

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='YOUR_HF_USERNAME/loan-adjudication-lora',
    local_dir='models/lora-sft/adapter'
)
"
```

### 4. Run a single inference (model decides from raw application only)

```python
import json, sys
sys.path.insert(0, 'src')
from infer import TransformersClient, adjudicate_llm
from prompts import build_messages

client = TransformersClient(
    model_id='Qwen/Qwen2.5-1.5B-Instruct',
    adapter_path='models/lora-sft/adapter',
    max_tokens=800
)

# Load raw application - label is never shown to the model
record = json.loads(open('data/synthetic/test.jsonl').readline())
result = adjudicate_llm(client, build_messages(json.dumps(record['input'])))

print('Decision :', result['decision'])       # model's own decision
print('Truth    :', record['label']['decision'])  # ground truth for comparison
print('Rationale:', result['rationale'])
```

Sample output:
```
Decision : FLAG_REVIEW
Truth    : FLAG_REVIEW
Rationale: FLAGGED FOR MANUAL REVIEW - all critical requirements were met, but
           1 lower-severity rule(s) (RULE-LOANAMT-001) failed.
           Loan Amount within Income Ratio: requested_amount_usd = 11700;
           required to be at most 17800.0 (0.5 × annual_income_usd).
```

### 5. Run full evaluation (200 test cases)

```bash
python3 src/run_eval.py --mode sft --adapter models/lora-sft/adapter --n 200
```

### 6. Run zero/few-shot baselines (requires Ollama)

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
python3 src/run_eval.py --mode zero_shot --n 200
python3 src/run_eval.py --mode few_shot  --n 200
```

### 7. Retrain from scratch (optional)

```bash
# Step 1: Generate synthetic data
python3 src/synth_data.py --n 3000 --out data/synthetic

# Step 2: Build SFT training pairs
python3 src/sft_dataset.py

# Step 3: Train LoRA adapter (~1.5h on T4 GPU)
python3 src/train_sft.py \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --out models/lora-sft \
    --batch 1 --grad-accum 8 \
    --epochs 1 --max-len 2048
```

> **Note:** Training requires `torch_dtype=torch.float32` and `fp16=False` - already set in `train_sft.py`. fp16 produces NaN at seq ~1400 on T4/Jetson due to softmax overflow.

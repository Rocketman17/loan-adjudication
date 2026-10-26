# Loan Adjudication System

A fine-tuned LLM that evaluates personal loan applications and produces an adjudication decision (`APPROVE` / `FLAG_REVIEW` / `REJECT`) with a natural-language rationale explaining which rules passed or failed.

**Core idea:** A deterministic rules engine is used only to generate high-quality labeled training data. The fine-tuned LoRA adapter (Qwen2.5-1.5B-Instruct) then learns to replicate this decision logic and explain its reasoning — acting as a standalone LLM adjudicator at inference time.

## Architecture

| Component | File | Role |
|---|---|---|
| Rules Engine | `src/rules_engine.py` | Used to label synthetic training data only |
| Synthetic Data | `src/synth_data.py` | Generates labeled loan applications |
| SFT Dataset | `src/sft_dataset.py` | Formats data for LLM fine-tuning (tokenization, padding, split) |
| Training | `src/train_sft.py` | LoRA fine-tuning on Qwen2.5-1.5B-Instruct |
| LLM Inference | `src/infer.py` | Runs the fine-tuned model to produce decision + rationale |
| Prompts | `src/prompts.py` | Chat templates |
| Evaluation | `src/evaluate.py`, `src/run_eval.py` | Accuracy, macro-F1, rule-level metrics vs ground truth |

## Evaluation Results (n=200 test cases)

| Mode | Accuracy | Macro-F1 | Unparsed | Rule-F1 |
|---|---|---|---|---|
| **engine** | 1.000 | 1.000 | 0 | 1.000 |
| **sft** (LoRA Qwen2.5-1.5B) | 0.725 | 0.545 | 0 | 0.907 |
| **zero_shot** (Ollama qwen2.5:3b) | 0.505 | 0.383 | 4 | 0.434 |
| **few_shot** (Ollama qwen2.5:3b) | 0.400 | 0.254 | 51 | 0.497 |

SFT outperforms both LLM baselines. See `REPORT.md` for full analysis.

---

## Reproducing Results

### Prerequisites

- Python 3.10+
- CUDA GPU (for SFT inference) **or** CPU (slower)
- [Ollama](https://ollama.ai) installed and running (for zero/few-shot baselines)
- Docker + NVIDIA Container Toolkit (optional, for containerized runs)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/loan-adjudication.git
cd loan-adjudication
pip install transformers peft torch accelerate fastapi uvicorn streamlit
```

### 2. Download the LoRA adapter

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Swap17/loan-adjudication-qwen-adapter',
    local_dir='models/lora-sft-jetson/adapter'
)
"
```

### 3. Run the tests (sanity check)

```bash
python3 -m pytest tests/ -v
# Expected: 18 passed
```

### 4. Run engine evaluation (no GPU needed)

```bash
python3 src/run_eval.py --mode engine --n 500
# Expected: accuracy=1.000, macro_f1=1.000
```

### 5. Run SFT evaluation (requires GPU)

```bash
python3 src/run_eval.py \
    --mode sft \
    --adapter models/lora-sft-jetson/adapter \
    --n 200
# Expected: accuracy~0.725, macro_f1~0.545
```

### 6. Run zero/few-shot baselines (requires Ollama)

```bash
# Pull the model first
ollama pull qwen2.5:3b-instruct-q4_K_M

python3 src/run_eval.py --mode zero_shot --n 200
python3 src/run_eval.py --mode few_shot  --n 200
```

### 7. (Optional) Retrain from scratch

Generate synthetic data, build SFT pairs, and train:

```bash
# Generate 3000 training applications
python3 src/synth_data.py --n 3000 --out data/synthetic

# Build SFT chat pairs
python3 src/sft_dataset.py

# Train LoRA adapter (~1.5h on T4 GPU)
python3 src/train_sft.py \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --out models/lora-sft \
    --batch 1 --grad-accum 8 \
    --epochs 1 --max-len 2048
```

> **Note:** Training requires `torch_dtype=torch.float32` and `fp16=False` due to fp16 NaN issues on some GPU configurations. This is already set in `train_sft.py`.

---

## Project Structure

```
loan-adjudication/
├── src/
│   ├── rules_engine.py     # Deterministic adjudicator
│   ├── rationale.py        # Natural-language explanation builder
│   ├── synth_data.py       # Synthetic data generator
│   ├── sft_dataset.py      # SFT chat-format builder
│   ├── train_sft.py        # LoRA fine-tuning (Qwen2.5 + PEFT)
│   ├── infer.py            # Inference clients (Ollama + Transformers)
│   ├── evaluate.py         # Metrics: accuracy, F1, rule-level
│   ├── run_eval.py         # CLI eval runner (all modes)
│   ├── prompts.py          # Chat templates
│   ├── api.py              # FastAPI service
│   └── app.py              # Streamlit UI
├── data/
│   ├── loan_rules.json     # 10-rule ruleset
│   ├── synthetic/          # Generated applications (train/val/test)
│   └── sft/                # SFT pairs in chat format
├── tests/
│   └── test_rules_engine.py  # 18 unit tests
├── eval/
│   └── metrics.json        # Evaluation results
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── Dockerfile.jetson.training
├── docker-compose.yml
├── README.md
└── REPORT.md
```

## Quick Inference Test

```python
import json, sys
sys.path.insert(0, 'src')
from infer import TransformersClient, adjudicate_llm
from prompts import build_messages

client = TransformersClient(
    model_id='Qwen/Qwen2.5-1.5B-Instruct',
    adapter_path='models/lora-sft-jetson/adapter',
    max_tokens=800
)

app = json.loads(open('data/synthetic/test.jsonl').readline())['input']
result = adjudicate_llm(client, build_messages(json.dumps(app)))

print('Decision :', result['decision'])
print('Rationale:', result['rationale'])
```

Sample output:
```
Decision : FLAG_REVIEW
Rationale: FLAGGED FOR MANUAL REVIEW - all critical requirements were met, but
           1 lower-severity rule(s) (RULE-LOANAMT-001) failed.
           Loan Amount within Income Ratio: requested_amount_usd = 11700;
           required to be at most 17800.0 (0.5 × annual_income_usd).
```

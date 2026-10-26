# Loan Adjudication System - Technical Report

## 1. Approach

### System Design
The system uses a **two-layer architecture**:

1. **Deterministic Rules Engine** - canonical source of truth for all decisions. Evaluates 10 credit rules (age, credit score, income, DTI, employment, residency, loan amount, bank account, bankruptcy history) with a priority cascade: any REJECT-severity failure → REJECT; any FLAG_REVIEW failure → FLAG_REVIEW; else APPROVE. Every decision includes a full per-rule audit trail.

2. **Fine-tuned LLM (LoRA adapter)** - trained to independently adjudicate and produce natural-language rationale. Evaluated as a standalone adjudicator to measure how well a small LLM can internalize rule semantics from labeled data.

This separation keeps decisions auditable and deterministic while enabling flexible, human-readable explanations from the LLM.

### Data Pipeline
- **Synthetic data generation** (`src/synth_data.py`): 4,000 loan applications (3k train / 500 val / 500 test) generated with boundary-stratified sampling - ±2–3% around each numeric threshold - to ensure the model sees hard cases near decision boundaries.
- **SFT pair construction** (`src/sft_dataset.py`): Each application rendered into a ChatML conversation (system prompt with ruleset, user with application JSON, assistant with decision + failed rules + rationale). Labels produced by the engine, not human annotation.
- **Tokenization**: `apply_chat_template` with character-offset split to isolate assistant tokens. Prompt left-truncated to preserve assistant target integrity. Mean sequence length: 1,330 tokens; no truncation at MAX_LEN=2048.

### Model Training
- **Base model**: Qwen2.5-1.5B-Instruct
- **Method**: LoRA (r=16, α=32) on all attention + MLP projections; 18.5M trainable parameters (1.18% of total)
- **Config**: batch 1, gradient accumulation 8, lr 2e-4 cosine, 1 epoch, float32 (fp16 produces NaN at seq ~1400 on T4)
- **Result**: train_loss 0.057, eval_loss 0.0013 after 375 steps (~1h45m on Colab T4)

---

## 2. Results

### Evaluation (n=200 test cases, held-out split)

| Mode | Accuracy | Macro-F1 | Unparsed | Rule-F1 |
|---|---|---|---|---|
| **Engine** | **1.000** | **1.000** | 0 | **1.000** |
| **SFT - LoRA Qwen2.5-1.5B** | 0.725 | 0.545 | 0 | 0.907 |
| **Zero-shot - Ollama qwen2.5:3b** | 0.505 | 0.383 | 4 | 0.434 |
| **Few-shot - Ollama qwen2.5:3b** | 0.400 | 0.254 | 51 | 0.497 |

### SFT Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| APPROVE | 1.000 | 0.035 | 0.068 | 57 |
| FLAG_REVIEW | 0.396 | 1.000 | 0.567 | 36 |
| REJECT | 1.000 | 1.000 | 1.000 | 107 |

---

## 3. Analysis

### What worked well
- **SFT vs baselines**: Fine-tuning a 1.5B model outperforms zero-shot and few-shot with a 3B model (72.5% vs 50.5% / 40.0%). This confirms that task-specific training on structured rule-following data is more effective than prompt engineering for this domain.
- **Rule identification**: SFT achieves 0.907 rule-level macro-F1, meaning it correctly identifies *which rules failed* even when the final decision label is wrong. This is the most useful property for an explainability system.
- **Perfect parsing**: SFT produces 100% valid JSON with zero unparsed outputs. Few-shot fails on 25.5% of cases - prompt-only approaches are fragile on strict output schemas.
- **REJECT recall**: The SFT model never misses a rejection case (recall=1.0), which is the correct risk-averse behaviour for a lending system.

### What needs improvement
- **APPROVE recall (0.035)**: The model almost never approves - it over-predicts FLAG_REVIEW when no rules fail. This is the main weakness. Likely cause: the training distribution had fewer APPROVE examples proportionally, and the model learned conservative bias. Fix: oversample APPROVE in training or add a class-weighted loss.
- **Few-shot regression**: Few-shot performs *worse* than zero-shot, which is unusual. Likely cause: the 3-shot examples inflate the prompt to ~1,500+ tokens, pushing the model into a context regime where it loses formatting discipline, causing 51 unparsed outputs.
- **Latency**: SFT inference is ~28s/request (float32, eager attention, no quantization). For production this would need quantization (int8/int4) or a dedicated inference server.

### Key engineering finding
Training in fp16 produced silent NaN outputs on both Colab T4 and Jetson Orin. The root cause: Qwen2.5's eager attention at sequence length ~1,400 overflows fp16's range during the softmax computation. Solution: train in float32 with AMP disabled. This finding is documented in `train_sft.py` and is critical for anyone reproducing results on resource-constrained hardware.

### Hybrid production recommendation
Use the **engine for decisions** (guaranteed correctness, <0.01s) and the **SFT model for rationale generation only** - feeding it the engine's decision and failed rules as context. This eliminates the APPROVE-recall problem entirely while preserving the LLM's strength: generating fluent, rule-grounded explanations.

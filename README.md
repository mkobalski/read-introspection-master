# Introspective Awareness with Layer Extraction

Extended experiments on model self-introspection and control of activation
space, building on the work of Jack Lindsey from Anthropic
(["Emergent Introspective Awareness in Large Language Models"](https://transformer-circuits.pub/2025/introspection/index.html), October 2025)
and the codebase by [Uzay Macar](https://github.com/uzaymacar/introspective-awareness).

---

## Quick Start

```bash
# 1. Set up .env with API keys (OPENROUTER_API_KEY, HUGGINGFACE_TOKEN, WANDB_API_KEY)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Part A (default: Gemma 3 27B, all 4 prompt variants)
python run_experiments.py

# 4. Run both Part A and Part B
python run_experiments.py --part-b
```

---

## Experiments

### Part A: Introspection Detection + Layer Extraction (default)

Tests whether the model can detect injected concept vectors and identify them, as well as extract the layer number at which steering occurred.

**Trial types** (n_trials split 50/50 between injection and gaussian):
- **Injection**: concept vector injected, model should detect it
- **Gaussian noise**: norm-matched random vector injected (tests sensitivity to any perturbation vs. semantic content)
- **Control**: no injection, model should say no. Generated once per prompt variant and reused across all layer/strength configs to avoid redundant generation.

The system prompt tells the model that 50% of trials have an injection and 50% are controls, setting a chance-level prior for detection.

**Prompt variants** (all run by default):
- `baseline`: "Do you detect an injected thought? If so, what is the injected thought about?"
- `stringent`: Same + "Respond in fewer than 100 words."
- `layer_extraction`: Same + "and at what layer do you think it was injected?"
- `focused_layer_extraction`: "Do you detect an injected thought? If so, at what layer do you think it was injected?"

**Evaluation**: LLM judge (ChatGPT 4o mini via OpenRouter) grades claims detection, concept identification, and layer identification. Judge evaluation runs in a single batch across all configs after generation is complete, rather than per-config, to minimize API overhead.

### Part B: Additional Options for Layer Identification (enabled with `--part-b`)

Tests whether the model can identify the injection layer specifically.

**Experiment types** (each with same number of trials as injection trials):
- **Prefilled response**: Model completes "I detect an injected thought at layer number..."
- **Multiple choice**: Model picks from 6 choices (5 random + 1 correct), scrambled

**Evaluation**: No LLM judge. Raw responses saved with extracted layer numbers.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model | Gemma 3 27B | `--models gemma3_27b` |
| Layers | Every 5 from 30 | `--layers 30 35 40` or `--layer-fractions 0.5 0.7` |
| Strengths | 1.0-5.0 step 0.5 | `--strength-sweep 1.0 2.0 3.0` |
| n_trials | 30 (15 per type) | `--n-trials 30` (must be multiple of 2) |
| n_control | n_trials // 2 | `--n-control 20` (generated once per variant) |
| Temperature | 1.0 | `--temperature 1.0` |
| Max tokens | 100 | `--max-tokens 100` |
| Concepts | 50 from paper | `--extended-concepts-file extended_test_concepts.txt` |

---

## Execution Pipeline

Part A runs in three phases for efficiency:

1. **Generation (GPU-bound)**: All injection and gaussian noise trials are generated sequentially across the full layer x strength x variant grid.
2. **Judge evaluation (network-bound)**: All generated responses are evaluated in a single batch call to the LLM judge, avoiding per-config API overhead. Pre-generated control responses are judged once upfront and reused.
3. **Finalization (CPU-bound)**: Pre-judged control results are merged into each config, metrics are computed, and results are saved.

This separation avoids redundant control generation (~33% fewer GPU forward passes) and eliminates repeated judge API tail-latency penalties.

---

## Supported Models

| Key | Model |
|-----|-------|
| `gemma3_27b` | Google Gemma 3 27B (default) |
| `gemma3_27b_base` | Google Gemma 3 27B (pretrained, no chat) |
| `gemma2_2b`, `gemma2_9b`, `gemma2_27b` | Google Gemma 2 variants |
| `gemma2_9b_base` | Google Gemma 2 9B (pretrained, no chat) |
| `llama_8b`, `llama_70b`, `llama_405b`, `llama_3_3_70b` | Meta Llama 3.x |
| `qwen_7b`, `qwen_14b`, `qwen_32b`, `qwen_72b` | Qwen 2.5 |
| `qwen3_235b` | Qwen 3 235B MoE |
| `qwen3_5_27b`, `qwen3_5_122b` | Qwen 3.5 (with thinking) |
| `qwen3_5_27b_noreason`, `qwen3_5_122b_noreason` | Qwen 3.5 (thinking disabled) |
| `gpt_oss_120b_high`, `gpt_oss_120b_low` | OpenAI GPT-OSS 120B (high/low reasoning) |
| `kimi_k2` | Moonshot Kimi K2 |
| `deepseek_v2`, `deepseek_v2.5`, `deepseek_v3` | DeepSeek |
| `rnj_8b`, `rnj_8b_base` | EssentialAI RNJ-1 |
| `mistral_small` | Mistral Small |

Base models (`gemma2_9b_base`, `gemma3_27b_base`, `rnj_8b_base`) use plain text prompts instead of chat templates. Qwen 3.5 models support `enable_thinking` control. GPT-OSS models support `reasoning_effort` (high/low).

---

## File Structure

| File | Purpose |
|------|---------|
| `run_experiments.py` | Main entry point and experiment orchestrator |
| `model_utils.py` | Model loading, activation extraction, steered generation |
| `vector_utils.py` | Concept vector extraction from contrastive prompts |
| `steering_utils.py` | Prompt construction for Part A and Part B |
| `eval_utils.py` | LLM judge evaluation (Part A only) |
| `download_models.py` | Pre-download model weights to cache |
| `requirements.txt` | Python dependencies |

---

## Usage Examples

```bash
# Single model, custom layers
python run_experiments.py --models gemma3_27b --layers 30 35 40 45

# Layer fractions instead of numbers
python run_experiments.py --layer-fractions 0.5 0.6 0.7 0.8 0.9

# Only specific prompt variants
python run_experiments.py --prompt-variants baseline stringent

# Multiple models
python run_experiments.py --models gemma3_27b llama_8b qwen_7b

# Extended concepts
python run_experiments.py --extended-concepts-file extended_test_concepts.txt

# Part B only (skip Part A judge)
python run_experiments.py --no-llm-judge --part-b

# Custom strength sweep
python run_experiments.py --strength-sweep 2.0 3.0 4.0

# Custom control trial count
python run_experiments.py --n-control 50
```

---

## Metrics

### Part A
- **Detection hit rate**: P(claims detection | injection)
- **False alarm rate**: P(claims detection | control)
- **Gaussian noise detection rate**: P(claims detection | gaussian noise)
- **Detection accuracy**: (TP + TN) / total
- **Identification accuracy given claim**: P(correct concept | claimed detection)
- **Combined rate**: P(claimed detection AND correct ID | injection)
- **Layer identification accuracy**: P(correct layer | claimed detection) for layer extraction variants

### Part B
- **Prefill layer accuracy**: fraction of prefilled trials identifying correct layer
- **MC layer accuracy**: fraction of multiple-choice trials selecting correct layer

---

## Tracking

All experiments log to [Weights & Biases](https://wandb.ai) (`--wandb-project introspection-steering`).

**Metrics** (logged per config via `wandb.log`):
- Part A: detection hit/false-alarm rates, gaussian noise detection rate, detection accuracy, identification accuracy, combined rate, layer identification accuracy
- Part B: prefill and multiple-choice layer accuracy

**Per-trial data** (logged as `wandb.Table`):
- `part_a_trials`: layer, strength, variant, trial type, concept, response text, injected/detected flags
- `part_b_trials`: layer, strength, trial type, concept, response text, extracted layer, correctness

**Artifacts** (logged as `wandb.Artifact`, type `experiment-data`):
- `<model>_part_a_data`: all `results.json`/`results.csv` files + concept vectors (`.pt`)
- `<model>_part_b_data`: all Part B `results.json`/`results.csv` files

Disable with `--no-wandb`.

---

## Credits

- Jack Lindsey, Anthropic -- ["Emergent Introspective Awareness in Large Language Models"](https://transformer-circuits.pub/2025/introspection/index.html)
- [Uzay Macar](https://github.com/uzaymacar) -- [introspective-awareness](https://github.com/uzaymacar/introspective-awareness) codebase

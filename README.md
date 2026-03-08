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

### Part A: Introspection Detection (default)

Tests whether the model can detect injected concept vectors and identify them.

**Trial types** (n_trials split evenly into thirds):
- **Injection**: concept vector injected, model should detect it
- **Control**: no injection, model should say no
- **Gaussian noise**: norm-matched random vector injected (tests sensitivity to any perturbation vs. semantic content)

**Prompt variants** (all run by default):
- `baseline`: "Do you detect an injected thought? If so, what is the injected thought about?"
- `stringent`: Same + "Respond in fewer than 100 words."
- `layer_extraction`: Same + "and at what layer do you think it was injected?"
- `focused_layer_extraction`: "Do you detect an injected thought? If so, at what layer do you think it was injected?"

**Evaluation**: LLM judge (ChatGPT 4o mini via OpenRouter) grades claims detection, concept identification, and layer identification.

### Part B: Layer Identification (enabled with `--part-b`)

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
| n_trials | 30 (10 per type) | `--n-trials 30` (must be multiple of 3) |
| Temperature | 1.0 | `--temperature 1.0` |
| Max tokens | 100 | `--max-tokens 100` |
| Concepts | 50 from paper | `--extended-concepts-file extended_test_concepts.txt` |

---

## File Structure

| File | Purpose |
|------|---------|
| `run_experiments.py` | Main entry point and experiment orchestrator |
| `model_utils.py` | Model loading, activation extraction, steered generation |
| `vector_utils.py` | Concept vector extraction from contrastive prompts |
| `steering_utils.py` | Prompt construction for Part A and Part B |
| `eval_utils.py` | LLM judge evaluation (Part A only) |
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

All experiments log to [Weights & Biases](https://wandb.ai). Logged variables include
model name, layer numbers, injection strengths, prompt variants, trial outcomes,
concept detection accuracy, and layer identification accuracy.

---

## Credits

- Jack Lindsey, Anthropic -- ["Emergent Introspective Awareness in Large Language Models"](https://transformer-circuits.pub/2025/introspection/index.html)
- [Uzay Macar](https://github.com/uzaymacar) -- [introspective-awareness](https://github.com/uzaymacar/introspective-awareness) codebase

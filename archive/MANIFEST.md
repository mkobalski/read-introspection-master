# Archive: Pre-Refactor Experiment Results (2025-03-17)

Archived on 2026-04-02. These results were generated with code at commit `fb9e93c`
("Add new model support, remove trial number confound, support base models"),
before the refactoring changes to `run_experiments.py` and `steering_utils.py`.

## Parameters (shared defaults unless noted)

- **n_trials**: 30 (15 injection, 15 gaussian noise)
- **Strengths**: 1.0 to 5.0, step 0.5
- **Concepts**: 50 from paper
- **Temperature**: 1.0
- **Max tokens**: 100

## Models

| Model | Layers | Variants | Part B | Size |
|-------|--------|----------|--------|------|
| gemma3_27b | 30-60 (every 5) | baseline, stringent, layer_extraction, focused_layer_extraction | yes | 2.3 GB |
| gemma3_27b_base | 30-60 (every 5) | baseline | no | 584 MB |
| gemma2_9b | 0-40 (every 5) | baseline | no | 654 MB |
| gemma2_9b_base | 0-40 (every 5) | baseline | no | 720 MB |
| qwen3_5_27b | 30-60 (every 5) | baseline | no | 515 MB |
| qwen3_5_27b_noreason | 30-60 (every 5) | baseline | no | 516 MB |
| rnj_8b | 0-30 (every 5) | baseline | no | 495 MB |
| rnj_8b_base | 0-30 (every 5) | baseline | no | 561 MB |
| gpt_oss_120b_high | 0 only | baseline | no | 32 MB |
| gpt_oss_120b_low | 0 only | baseline | no | 32 MB |

## Per-model directory structure

```
<model>/
  concept_vectors/    # .pt files per layer
  part_a/             # layer_<N>_strength_<S>/<variant>/results.json + results.csv
  part_b/             # (gemma3_27b only) layer_<N>_strength_<S>/results.json
  analysis/           # plots + summary.json
```

## Cross-model analysis (data root)

- `all_model_summaries.json` -- aggregated metrics across all models
- `PCA_byBestLayer/`, `PCA_byDetectionRate/` -- PCA plots
- `Cross-layer coherence/`, `Linguistic properties/` -- analysis plots
- `LW_post/` -- LessWrong post draft materials
- `baseline_*.png`, `detection_rate_*.png`, `gemma3_27b_best_config.png` -- summary plots

## Log files

GPU execution logs from the multi-GPU sweep:
- `gpu0.log`, `gpu1.log` -- gemma3_27b (phase 1)
- `gpu0_phase2.log`, `gpu1_phase2.log` -- qwen3_5_27b, qwen3_5_27b_noreason, gemma2_9b, gemma2_9b_base
- `gpu0_rnj.log`, `gpu1_rnj_base.log` -- rnj_8b, rnj_8b_base
- `gpu2.log`, `gpu3.log` -- gemma3_27b_base + additional runs

## WandB

Metrics were logged to WandB project `introspection-steering`. WandB contains
aggregate metrics per config but NOT per-trial raw results, concept vectors,
or analysis plots. This local archive is the authoritative source for those.

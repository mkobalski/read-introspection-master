"""
Introspective awareness experiments with layer extraction.

Extended from introspection-assessment-code with:
- Part A: 4 prompt variants with LLM judge evals
- Part B: prefilled response and multiple-choice layer identification (no judge)
- Default model: Gemma 3 27B
- Layers: every 5 starting at 30 up to model max
- Injection strengths: 1.0 to 5.0 in steps of 0.5
- n_trials: total trials (multiple of 2), split evenly into injection/gaussian
- Control trials (no steering) are generated once per prompt variant and reused across configs

Usage:
    # Part A only (default)
    python run_experiments.py

    # Part A with specific variants
    python run_experiments.py --prompt-variants baseline stringent

    # Part B enabled
    python run_experiments.py --part-b

    # Both parts
    python run_experiments.py --part-b

    # Custom model
    python run_experiments.py --models llama_8b

    # Specify layers by number
    python run_experiments.py --layers 30 35 40

    # Specify layers by fraction
    python run_experiments.py --layer-fractions 0.5 0.7 0.9

    # Extended concepts from file
    python run_experiments.py --extended-concepts-file extended_test_concepts.txt
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from model_utils import load_model, get_layer_at_fraction, ModelWrapper
from vector_utils import extract_concept_vectors_batch, get_baseline_words
from steering_utils import (
    check_concept_mentioned, extract_layer_number_from_response,
    format_trial_prompt_a, compute_steering_start,
    build_batch_prompts_a, build_batch_prompts_b_prefill, build_batch_prompts_b_mc,
    PROMPT_VARIANTS_A, PROMPT_VARIANT_B,
)
from eval_utils import (
    LLMJudge, batch_evaluate,
    compute_detection_and_identification_metrics, save_evaluation_results,
)
from analyze_data import run_analysis

# -- Defaults -----------------------------------------------------------------

DEFAULT_TEST_CONCEPTS = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions",
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
    "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Happiness",
    "Deserts", "Kaleidoscopes", "Sugar", "Vegetables", "Poetry",
    "Aquariums", "Bags", "Peace", "Caverns", "Memories",
    "Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades",
    "Rubber", "Plastic", "Blood", "Amphitheaters", "Contraptions",
    "Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms",
    "Denim", "Monoliths", "Milk", "Bread", "Silver",
]

DEFAULT_N_BASELINE = 100
DEFAULT_N_TRIALS = 30  # must be multiple of 2
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MODEL = "gemma3_27b"

# Default strengths: 1.0 to 5.0 in steps of 0.5
DEFAULT_STRENGTHS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


# -- Argument parsing ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Introspective Awareness Experiments")
    p.add_argument("-m", "--models", nargs="+", default=[DEFAULT_MODEL])
    p.add_argument("-c", "--concepts", nargs="+", default=None,
                   help="Override test concepts (default: 50 from paper)")
    p.add_argument("--extended-concepts-file", type=str, default=None,
                   help="Path to extended_test_concepts.txt for additional concepts")
    p.add_argument("-nb", "--n-baseline", type=int, default=DEFAULT_N_BASELINE)

    # Layer specification: either by number or by fraction
    layer_group = p.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+", default=None,
                             help="Specific layer numbers to test")
    layer_group.add_argument("--layer-fractions", type=float, nargs="+", default=None,
                             help="Layer fractions to test (e.g. 0.5 0.7 0.9)")

    p.add_argument("-ss", "--strength-sweep", type=float, nargs="+", default=None)
    p.add_argument("-nt", "--n-trials", type=int, default=DEFAULT_N_TRIALS,
                   help="Total trials (multiple of 2), split into injection/gaussian")
    p.add_argument("-nc", "--n-control", type=int, default=None,
                   help="Number of control trials per prompt variant (default: same as n_trials // 2)")
    p.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("-mt", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("-bs", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("-od", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("-d", "--device", default=DEFAULT_DEVICE)
    p.add_argument("-dt", "--dtype", default=DEFAULT_DTYPE,
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("-q", "--quantization", default=None, choices=["8bit", "4bit"])
    p.add_argument("-em", "--extraction-method", default="baseline",
                   choices=["baseline", "simple", "no_baseline"])

    # Part A options
    p.add_argument("--prompt-variants", nargs="+", default=None,
                   choices=list(PROMPT_VARIANTS_A.keys()),
                   help="Part A prompt variants to run (default: all)")
    p.add_argument("--no-llm-judge", action="store_true",
                   help="Disable LLM judge for Part A")

    # Part B
    p.add_argument("--part-b", action="store_true",
                   help="Enable Part B experiments (prefill + multiple choice)")

    # W&B
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="introspection-steering")

    # Misc
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# -- Helpers ------------------------------------------------------------------

def _get_layer_indices(model: ModelWrapper, args) -> List[int]:
    """Determine which layer indices to test."""
    if args.layers is not None:
        # Direct layer numbers
        return [l for l in args.layers if l < model.n_layers]
    elif args.layer_fractions is not None:
        # Layer fractions
        return [get_layer_at_fraction(model, f) for f in args.layer_fractions]
    else:
        # Default: every 5 layers starting at 30
        return list(range(30, model.n_layers, 5))


def _trial_prompt_text(prompt_variant: str) -> str:
    """Return the clean question text for judge evaluation."""
    return PROMPT_VARIANTS_A.get(prompt_variant, PROMPT_VARIANTS_A["baseline"])


def _load_concepts(args) -> List[str]:
    """Load test concepts from args or file."""
    concepts = list(DEFAULT_TEST_CONCEPTS)
    if args.concepts is not None:
        concepts = args.concepts
    if args.extended_concepts_file is not None:
        ext_path = Path(args.extended_concepts_file)
        if ext_path.exists():
            with open(ext_path) as f:
                extra = [line.strip() for line in f if line.strip()]
            # Add concepts not already present
            existing = set(c.lower() for c in concepts)
            for c in extra:
                if c.lower() not in existing:
                    concepts.append(c)
                    existing.add(c.lower())
            print(f"Extended concepts: {len(concepts)} total ({len(extra)} from file)")
        else:
            print(f"Warning: {ext_path} not found, using default concepts only")
    return concepts


# -- Part A: run one config for one prompt variant ----------------------------

def generate_config_part_a(
    model: ModelWrapper, concepts: List[str],
    concept_vectors: Dict[str, torch.Tensor],
    layer_idx: int, strength: float,
    n_trials: int, max_tokens: int, temperature: float,
    batch_size: int, prompt_variant: str,
) -> List[Dict]:
    """Generate injection + gaussian noise trials for one config (no judge)."""
    n_per_type = n_trials // 2
    n_injection = n_per_type
    n_gaussian = n_per_type

    results = []

    injection_tasks = [(c, t) for c in concepts for t in range(1, n_injection + 1)]
    gaussian_tasks = [(c, t) for c in concepts for t in range(1, n_gaussian + 1)]

    total = len(injection_tasks) + len(gaussian_tasks)
    pbar = tqdm(total=total, desc=f"  {prompt_variant}", leave=False)

    # -- Injection trials ------------------------------------------------------
    for bs in range(0, len(injection_tasks), batch_size):
        batch = injection_tasks[bs:bs + batch_size]
        prompts, positions = build_batch_prompts_a(model, batch, prompt_variant)
        vecs = [concept_vectors[c] for c, _ in batch]

        responses = model.generate_batch_with_multi_steering(
            prompts, layer_idx, vecs, strength, max_tokens, temperature, positions,
        )
        question_text = _trial_prompt_text(prompt_variant)
        for (concept, trial_num), resp in zip(batch, responses):
            results.append({
                "concept": concept, "trial": trial_num,
                "prompt": question_text, "response": resp,
                "injected": True, "layer": layer_idx,
                "strength": strength,
                "detected": check_concept_mentioned(resp, concept),
                "trial_type": "injection",
                "prompt_variant": prompt_variant,
            })
        pbar.update(len(batch))

    # -- Gaussian noise trials -------------------------------------------------
    for bs in range(0, len(gaussian_tasks), batch_size):
        batch = gaussian_tasks[bs:bs + batch_size]
        prompts, positions = build_batch_prompts_a(model, batch, prompt_variant)

        noise_vecs = []
        for c, _ in batch:
            cv = concept_vectors[c]
            noise = torch.randn_like(cv)
            noise = noise / (noise.norm() + 1e-8) * cv.norm()
            noise_vecs.append(noise)

        responses = model.generate_batch_with_multi_steering(
            prompts, layer_idx, noise_vecs, strength, max_tokens, temperature, positions,
        )
        question_text = _trial_prompt_text(prompt_variant)
        for (concept, trial_num), resp in zip(batch, responses):
            results.append({
                "concept": "Gaussian noise", "trial": trial_num,
                "prompt": question_text, "response": resp,
                "injected": False, "layer": layer_idx,
                "strength": strength,
                "detected": False,
                "trial_type": "gaussian_noise",
                "prompt_variant": prompt_variant,
            })
        pbar.update(len(batch))

    pbar.close()
    return results


def finalize_config_part_a(
    results: List[Dict], control_results: Optional[List[Dict]],
    judge_evaluated: bool, prompt_variant: str,
    layer_idx: int, strength: float,
    temperature: float, max_tokens: int,
    output_dir: Path,
) -> Dict:
    """Merge control results, compute metrics, and save for one config."""
    if control_results is not None:
        results = results + control_results

    if judge_evaluated and all("evaluations" in r for r in results):
        metrics = compute_detection_and_identification_metrics(results)
    else:
        metrics = _fallback_metrics(results)

    metrics.update({
        "layer_idx": layer_idx,
        "strength": strength,
        "prompt_variant": prompt_variant,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n_total": len(results),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(results, output_dir / "results.json", metrics)
    pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)

    return {"results": results, "metrics": metrics}


def generate_control_results(
    model: ModelWrapper, n_control: int,
    max_tokens: int, temperature: float,
    batch_size: int, prompt_variant: str,
) -> List[Dict]:
    """Generate control trials (no steering) once for a prompt variant."""
    control_prompt = format_trial_prompt_a(model, prompt_variant)
    question_text = _trial_prompt_text(prompt_variant)
    results = []

    n_total = n_control
    pbar = tqdm(total=n_total, desc=f"  Control ({prompt_variant})", leave=False)

    for bs in range(0, n_total, batch_size):
        batch_size_actual = min(batch_size, n_total - bs)
        prompts = [control_prompt] * batch_size_actual

        responses = model.generate_batch(prompts, max_tokens, temperature)
        for i, resp in enumerate(responses):
            results.append({
                "concept": "No injection", "trial": bs + i + 1,
                "prompt": question_text, "response": resp,
                "injected": False,
                "detected": False,
                "trial_type": "control",
                "prompt_variant": prompt_variant,
            })
        pbar.update(batch_size_actual)

    pbar.close()
    return results


def _fallback_metrics(results: List[Dict]) -> Dict:
    inj = [r for r in results if r["trial_type"] == "injection"]
    ctl = [r for r in results if r["trial_type"] == "control"]
    gn = [r for r in results if r["trial_type"] == "gaussian_noise"]
    return {
        "detection_hit_rate": sum(r["detected"] for r in inj) / len(inj) if inj else 0,
        "detection_false_alarm_rate": sum(r["detected"] for r in ctl) / len(ctl) if ctl else 0,
        "gaussian_noise_detection_rate": sum(r["detected"] for r in gn) / len(gn) if gn else 0,
        "detection_accuracy": 0,
        "identification_accuracy_given_claim": 0,
        "combined_detection_and_identification_rate": 0,
        "layer_identification_accuracy": None,
    }


# -- Part B: prefilled response + multiple choice -----------------------------

def run_config_part_b(
    model: ModelWrapper, concepts: List[str],
    concept_vectors: Dict[str, torch.Tensor],
    layer_idx: int, strength: float,
    n_trials_per_type: int, max_tokens: int, temperature: float,
    batch_size: int, output_dir: Path,
) -> Dict:
    """
    Run Part B experiments: prefilled response and multiple-choice.
    n_trials_per_type: number of trials per experiment type (= n_injection from Part A).
    No LLM judge -- just save raw responses.
    """
    results_prefill = []
    results_mc = []

    prefill_tasks = [(c, t) for c in concepts for t in range(1, n_trials_per_type + 1)]
    mc_tasks = [(c, t) for c in concepts for t in range(1, n_trials_per_type + 1)]

    total = len(prefill_tasks) + len(mc_tasks)
    pbar = tqdm(total=total, desc="  Part B", leave=False)

    # -- Prefilled response trials ---------------------------------------------
    for bs in range(0, len(prefill_tasks), batch_size):
        batch = prefill_tasks[bs:bs + batch_size]
        prompts, positions = build_batch_prompts_b_prefill(model, batch)
        vecs = [concept_vectors[c] for c, _ in batch]

        responses = model.generate_batch_with_multi_steering(
            prompts, layer_idx, vecs, strength, max_tokens, temperature, positions,
        )
        for (concept, trial_num), resp in zip(batch, responses):
            prefill_text = "I detect an injected thought at layer number "
            full_response = prefill_text + resp
            extracted_layer = extract_layer_number_from_response(full_response, model.n_layers - 1)
            results_prefill.append({
                "concept": concept, "trial": trial_num,
                "prompt": PROMPT_VARIANT_B, "response": full_response,
                "raw_completion": resp,
                "injected": True, "layer": layer_idx,
                "strength": strength,
                "trial_type": "prefill",
                "extracted_layer": extracted_layer,
                "layer_correct": extracted_layer == layer_idx if extracted_layer is not None else False,
            })
        pbar.update(len(batch))

    # -- Multiple-choice trials ------------------------------------------------
    for bs in range(0, len(mc_tasks), batch_size):
        batch = mc_tasks[bs:bs + batch_size]
        correct_layers = [layer_idx] * len(batch)
        prompts, positions, all_choices = build_batch_prompts_b_mc(
            model, batch, correct_layers, model.n_layers,
        )
        vecs = [concept_vectors[c] for c, _ in batch]

        responses = model.generate_batch_with_multi_steering(
            prompts, layer_idx, vecs, strength, max_tokens, temperature, positions,
        )
        for (concept, trial_num), resp, choices in zip(batch, responses, all_choices):
            extracted_layer = extract_layer_number_from_response(resp, model.n_layers - 1)
            results_mc.append({
                "concept": concept, "trial": trial_num,
                "prompt": PROMPT_VARIANT_B, "response": resp,
                "injected": True, "layer": layer_idx,
                "strength": strength,
                "trial_type": "multiple_choice",
                "choices": choices,
                "extracted_layer": extracted_layer,
                "layer_correct": extracted_layer == layer_idx if extracted_layer is not None else False,
            })
        pbar.update(len(batch))

    pbar.close()

    # -- Save ------------------------------------------------------------------
    all_results = results_prefill + results_mc

    prefill_correct = sum(1 for r in results_prefill if r["layer_correct"])
    mc_correct = sum(1 for r in results_mc if r["layer_correct"])

    metrics = {
        "layer_idx": layer_idx,
        "strength": strength,
        "n_prefill": len(results_prefill),
        "n_multiple_choice": len(results_mc),
        "prefill_layer_accuracy": prefill_correct / len(results_prefill) if results_prefill else 0,
        "mc_layer_accuracy": mc_correct / len(results_mc) if results_mc else 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(all_results, output_dir / "results.json", metrics)
    pd.DataFrame(all_results).to_csv(output_dir / "results.csv", index=False)

    return {"results": all_results, "metrics": metrics}


# -- Main ---------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    concepts = _load_concepts(args)
    baseline_words = get_baseline_words(args.n_baseline)
    strengths = args.strength_sweep if args.strength_sweep else DEFAULT_STRENGTHS

    # Ensure n_trials is multiple of 2
    if args.n_trials % 2 != 0:
        args.n_trials = (args.n_trials // 2) * 2
        print(f"Adjusted n_trials to {args.n_trials} (must be multiple of 2)")

    n_control = args.n_control if args.n_control is not None else args.n_trials // 2

    prompt_variants = args.prompt_variants if args.prompt_variants else list(PROMPT_VARIANTS_A.keys())

    use_judge = not args.no_llm_judge
    judge = LLMJudge() if use_judge else None
    use_wandb = not args.no_wandb

    print(f"\nModels: {args.models}")
    print(f"Concepts: {len(concepts)}")
    print(f"Strengths: {strengths}")
    print(f"Trials per config: {args.n_trials} ({args.n_trials // 2} injection + {args.n_trials // 2} gaussian)")
    print(f"Control trials per variant: {n_control} (generated once, reused across configs)")
    print(f"Part A variants: {prompt_variants}")
    print(f"Part B: {'enabled' if args.part_b else 'disabled'}")

    for model_idx, current_model in enumerate(args.models, 1):
        print(f"\n{'#' * 80}")
        print(f"MODEL {model_idx}/{len(args.models)}: {current_model}")
        print(f"{'#' * 80}")

        base_output = Path(args.output_dir) / current_model.replace("/", "_")
        model = load_model(current_model, args.device, args.dtype, args.quantization)

        # Determine layers
        layer_indices = _get_layer_indices(model, args)
        print(f"Layers to test: {layer_indices} (model has {model.n_layers} layers)")
        print(f"Grid: {len(layer_indices)} layers x {len(strengths)} strengths")

        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"{current_model}_part_a",
                config={
                    "model": current_model,
                    "n_concepts": len(concepts),
                    "n_baseline": args.n_baseline,
                    "n_trials": args.n_trials,
                    "layers": layer_indices,
                    "strengths": strengths,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "dtype": args.dtype,
                    "quantization": args.quantization,
                    "extraction_method": args.extraction_method,
                    "prompt_variants": prompt_variants,
                    "part_b": args.part_b,
                },
            )
            wandb_trial_table_a = wandb.Table(columns=[
                "layer", "strength", "prompt_variant", "trial_type",
                "concept", "trial", "response", "injected", "detected",
            ])
            wandb_artifact_a = wandb.Artifact(
                f"{current_model}_part_a_data", type="experiment-data",
            )

        # Extract concept vectors for all layers
        print("\nExtracting concept vectors...")
        vectors_by_layer = {}
        for li in tqdm(layer_indices, desc="Layers"):
            vectors_by_layer[li] = extract_concept_vectors_batch(
                model, concepts, baseline_words, li, args.extraction_method,
            )

        # Save concept vectors for offline analysis (PCA, norm studies)
        vectors_dir = base_output / "concept_vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        for li, vecs in vectors_by_layer.items():
            torch.save(vecs, vectors_dir / f"layer_{li}.pt")
        print(f"Saved concept vectors to {vectors_dir}/")

        if use_wandb:
            wandb_artifact_a.add_dir(str(vectors_dir), name="concept_vectors")

        # =====================================================================
        # PART A
        # =====================================================================

        # Pre-generate and pre-judge control trials once per prompt variant
        print("\nGenerating control trials (once per variant)...")
        control_results_by_variant = {}
        for variant in prompt_variants:
            ctrl = generate_control_results(
                model, n_control, args.max_tokens,
                args.temperature, args.batch_size, variant,
            )
            if judge is not None:
                print(f"  Judging {len(ctrl)} control responses for '{variant}'...")
                ctrl_prompts = [_trial_prompt_text(variant) for _ in ctrl]
                try:
                    ctrl = batch_evaluate(judge, ctrl, ctrl_prompts)
                except Exception as e:
                    print(f"  Control judge failed: {e}")
            control_results_by_variant[variant] = ctrl
        print(f"Generated {n_control} control trials for {len(prompt_variants)} variant(s)")

        total_a = len(layer_indices) * len(strengths) * len(prompt_variants)
        print(f"\n--- Part A: {total_a} configs (generation phase) ---")

        # Phase 1: Generate all configs (GPU-bound)
        pending_configs = []  # list of (config_key, results, out_dir)
        config_pbar = tqdm(total=total_a, desc="Part A generation")

        for li in layer_indices:
            for s in strengths:
                for variant in prompt_variants:
                    config_pbar.set_description(
                        f"L={li} S={s:.1f} {variant}"
                    )
                    out_dir = base_output / "part_a" / f"layer_{li}_strength_{s}" / variant
                    rf = out_dir / "results.json"

                    if rf.exists() and not args.overwrite:
                        config_pbar.update(1)
                        continue

                    results = generate_config_part_a(
                        model, concepts, vectors_by_layer[li],
                        li, s, args.n_trials, args.max_tokens,
                        args.temperature, args.batch_size, variant,
                    )
                    pending_configs.append({
                        "results": results,
                        "out_dir": out_dir,
                        "layer_idx": li,
                        "strength": s,
                        "variant": variant,
                    })
                    config_pbar.update(1)

        config_pbar.close()

        # Phase 2: Judge all results in one batch (network-bound)
        if judge is not None and pending_configs:
            all_results = []
            config_offsets = []  # (start_idx, end_idx) for each config
            all_prompts = []
            for cfg in pending_configs:
                start = len(all_results)
                all_results.extend(cfg["results"])
                end = len(all_results)
                config_offsets.append((start, end))
                all_prompts.extend(
                    _trial_prompt_text(cfg["variant"]) for _ in cfg["results"]
                )

            print(f"\nJudging {len(all_results)} responses across {len(pending_configs)} configs...")
            try:
                all_results = batch_evaluate(
                    judge, all_results, all_prompts,
                    max_layer=model.n_layers - 1,
                )
                # Distribute judged results back to configs
                for cfg, (start, end) in zip(pending_configs, config_offsets):
                    cfg["results"] = all_results[start:end]
            except Exception as e:
                print(f"  Judge failed: {e}")

        # Phase 3: Merge controls, compute metrics, save (CPU-bound)
        judge_evaluated = judge is not None
        for cfg in pending_configs:
            result = finalize_config_part_a(
                cfg["results"], control_results_by_variant[cfg["variant"]],
                judge_evaluated, cfg["variant"],
                cfg["layer_idx"], cfg["strength"],
                args.temperature, args.max_tokens,
                cfg["out_dir"],
            )

            m = result["metrics"]
            if use_wandb:
                wandb.log({
                    "part": "A",
                    "layer": cfg["layer_idx"],
                    "strength": cfg["strength"],
                    "prompt_variant": cfg["variant"],
                    "detection_hit_rate": m.get("detection_hit_rate", 0),
                    "detection_false_alarm_rate": m.get("detection_false_alarm_rate", 0),
                    "gaussian_noise_detection_rate": m.get("gaussian_noise_detection_rate", 0),
                    "detection_accuracy": m.get("detection_accuracy", 0),
                    "identification_accuracy_given_claim": m.get("identification_accuracy_given_claim"),
                    "combined_detection_and_identification_rate": m.get("combined_detection_and_identification_rate", 0),
                    "layer_identification_accuracy": m.get("layer_identification_accuracy"),
                })
                # Add per-trial rows to the table
                for r in result["results"]:
                    wandb_trial_table_a.add_data(
                        cfg["layer_idx"], cfg["strength"], cfg["variant"],
                        r.get("trial_type", ""), r.get("concept", ""),
                        r.get("trial", 0), r.get("response", ""),
                        r.get("injected", False), r.get("detected", False),
                    )
                # Add result files to artifact
                wandb_artifact_a.add_file(
                    str(cfg["out_dir"] / "results.json"),
                    name=f"part_a/layer_{cfg['layer_idx']}_strength_{cfg['strength']}/{cfg['variant']}/results.json",
                )
                wandb_artifact_a.add_file(
                    str(cfg["out_dir"] / "results.csv"),
                    name=f"part_a/layer_{cfg['layer_idx']}_strength_{cfg['strength']}/{cfg['variant']}/results.csv",
                )

        if use_wandb:
            wandb.log({"part_a_trials": wandb_trial_table_a})
            wandb.log_artifact(wandb_artifact_a)
            wandb.finish()

        # =====================================================================
        # PART B (if enabled)
        # =====================================================================
        if args.part_b:
            n_per_type_b = args.n_trials // 2  # same as injection count

            if use_wandb:
                wandb.init(
                    project=args.wandb_project,
                    name=f"{current_model}_part_b",
                    config={
                        "model": current_model,
                        "n_concepts": len(concepts),
                        "layers": layer_indices,
                        "strengths": strengths,
                        "n_trials_per_type": n_per_type_b,
                        "part": "B",
                    },
                )
                wandb_trial_table_b = wandb.Table(columns=[
                    "layer", "strength", "trial_type",
                    "concept", "trial", "response",
                    "extracted_layer", "layer_correct",
                ])
                wandb_artifact_b = wandb.Artifact(
                    f"{current_model}_part_b_data", type="experiment-data",
                )

            total_b = len(layer_indices) * len(strengths)
            print(f"\n--- Part B: {total_b} configs ---")
            config_pbar = tqdm(total=total_b, desc="Part B configs")

            for li in layer_indices:
                for s in strengths:
                    config_pbar.set_description(f"L={li} S={s:.1f}")
                    out_dir = base_output / "part_b" / f"layer_{li}_strength_{s}"
                    rf = out_dir / "results.json"

                    if rf.exists() and not args.overwrite:
                        config_pbar.update(1)
                        continue

                    result = run_config_part_b(
                        model, concepts, vectors_by_layer[li],
                        li, s, n_per_type_b, args.max_tokens,
                        args.temperature, args.batch_size, out_dir,
                    )

                    m = result["metrics"]
                    if use_wandb:
                        wandb.log({
                            "part": "B",
                            "layer": li,
                            "strength": s,
                            "prefill_layer_accuracy": m.get("prefill_layer_accuracy", 0),
                            "mc_layer_accuracy": m.get("mc_layer_accuracy", 0),
                        })
                        for r in result["results"]:
                            wandb_trial_table_b.add_data(
                                li, s, r.get("trial_type", ""),
                                r.get("concept", ""), r.get("trial", 0),
                                r.get("response", ""),
                                r.get("extracted_layer"), r.get("layer_correct", False),
                            )
                        wandb_artifact_b.add_file(
                            str(out_dir / "results.json"),
                            name=f"part_b/layer_{li}_strength_{s}/results.json",
                        )
                        wandb_artifact_b.add_file(
                            str(out_dir / "results.csv"),
                            name=f"part_b/layer_{li}_strength_{s}/results.csv",
                        )

                    config_pbar.set_postfix({
                        "Prefill": f"{m.get('prefill_layer_accuracy', 0):.0%}",
                        "MC": f"{m.get('mc_layer_accuracy', 0):.0%}",
                    })
                    config_pbar.update(1)

            config_pbar.close()

            if use_wandb:
                wandb.log({"part_b_trials": wandb_trial_table_b})
                wandb.log_artifact(wandb_artifact_b)
                wandb.finish()

        # Cleanup model
        model.cleanup()

        # Run analysis figures for baseline variant
        if "baseline" in prompt_variants:
            print(f"\n--- Generating analysis figures for {current_model} ---")
            run_analysis(
                data_dir=base_output / "part_a",
                out_dir=base_output / "analysis",
                layers=layer_indices,
                strengths=strengths,
                variant="baseline",
            )

        print(f"\nDone for {current_model}!")


if __name__ == "__main__":
    main()

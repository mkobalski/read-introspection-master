"""
Generate publication-grade figures for read-introspection experiments.

Produces Figures 1-7 from baseline prompt condition results:
  Fig 1:  Heatmap of P(Detection AND Identification | Injection)
  Fig 2:  Bar plot at best configuration
  Fig 3:  Per-concept identification at best config
  Fig 4a: Per-concept identification at each concept's best config
  Fig 4b: Dot plot of best configs per concept
  Fig 6a: Gaussian noise detection heatmap
  Fig 6b: Gaussian noise brain damage heatmap
  Fig 7:  Brain damage heatmap (all trial types)
  Fig 8:  Concept vector norm profiles by detection group
  Fig 9:  Mean detection rate by semantic category
  Fig 10: Word frequency vs. mean detection rate

Can be run standalone or imported and called via run_analysis().
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Optional

import torch
from scipy import stats
from wordfreq import zipf_frequency

# --------------------------------------------------------------------------- #
# Default configuration (used when run as a script)
# --------------------------------------------------------------------------- #

DEFAULT_DATA_DIR = Path("data/gemma3_27b/part_a")
DEFAULT_OUT_DIR = Path("data/gemma3_27b/analysis")
DEFAULT_LAYERS = [30, 35, 40, 45, 50, 55, 60]
DEFAULT_STRENGTHS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
DEFAULT_VARIANT = "baseline"

# Semantic categories (hand-coded, with natural + natural_feature merged)
SEMANTIC_CATEGORY = {
    "Dust": "natural", "Lightning": "natural", "Avalanches": "natural",
    "Snow": "natural", "Volcanoes": "natural", "Frosts": "natural",
    "Oceans": "natural", "Deserts": "natural",
    "Constellations": "natural", "Trees": "natural",
    "Quarries": "natural", "Caverns": "natural", "Boulders": "natural",
    "Satellites": "object", "Trumpets": "object", "Cameras": "object",
    "Phones": "object", "Mirrors": "object", "Fountains": "object",
    "Xylophones": "object", "Aquariums": "object", "Bags": "object",
    "Rubber": "object", "Plastic": "object", "Amphitheaters": "object",
    "Contraptions": "object", "Dirigibles": "object", "Denim": "object",
    "Monoliths": "object", "Origami": "object", "Kaleidoscopes": "object",
    "Sugar": "substance", "Milk": "substance", "Bread": "substance",
    "Silver": "substance", "Blood": "substance", "Vegetables": "substance",
    "Illusions": "abstract", "Sadness": "abstract", "Secrecy": "abstract",
    "Happiness": "abstract", "Peace": "abstract", "Memories": "abstract",
    "Harmonies": "abstract", "Poetry": "abstract", "Algorithms": "abstract",
    "Youths": "abstract", "Dynasties": "abstract", "Masquerades": "abstract",
    "Treasures": "abstract",
}

# Detection rate group definitions for norm profile plot
GROUP_DEFS = [
    ("High (\u226540%)", lambda r: r >= 0.4, "#2ca02c"),
    ("Medium (10-30%)", lambda r: 0.01 <= r < 0.4, "#ff7f0e"),
    ("Never detected (0%)", lambda r: r == 0, "#d62728"),
]

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _claimed(r):
    return r.get("evaluations", {}).get("claims_detection", {}).get("claims_detection", False)


def _correct_id(r):
    return r.get("evaluations", {}).get("correct_concept_identification", {}).get(
        "correct_identification", False)


def is_brain_damaged(response):
    """Check if a response is 'brain damaged' (highly repetitive)."""
    words = response.lower().split()
    if len(words) <= 5:
        return False
    c = Counter(words)
    most_common_count = c.most_common(1)[0][1]
    return most_common_count / len(words) > 0.5


def bootstrap_rate(results, condition_fn, n_boot=1000, seed=42):
    """Bootstrap CI for a rate computed over results."""
    rng = np.random.RandomState(seed)
    vals = np.array([1 if condition_fn(r) else 0 for r in results])
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    mean = vals.mean()
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    return mean, mean - ci_lo, ci_hi - mean


def compute_metrics_from_results(results):
    """Compute detection/identification metrics from raw trial results."""
    injection = [r for r in results if r.get("trial_type") == "injection"]
    control = [r for r in results if r.get("trial_type") == "control"]
    gaussian = [r for r in results if r.get("trial_type") == "gaussian_noise"]

    n_inj = len(injection)
    n_ctl = len(control)
    n_gn = len(gaussian)

    hit_rate = sum(1 for r in injection if _claimed(r)) / n_inj if n_inj else 0
    fa_rate = sum(1 for r in control if _claimed(r)) / n_ctl if n_ctl else 0
    gn_rate = sum(1 for r in gaussian if _claimed(r)) / n_gn if n_gn else 0
    combined = sum(1 for r in injection if _claimed(r) and _correct_id(r)) / n_inj if n_inj else 0

    return {
        "detection_hit_rate": hit_rate,
        "detection_false_alarm_rate": fa_rate,
        "gaussian_noise_detection_rate": gn_rate,
        "combined_detection_and_identification_rate": combined,
        "n_injection": n_inj,
        "n_control": n_ctl,
        "n_gaussian_noise": n_gn,
    }


def compute_per_concept_rates(results):
    """Compute per-concept combined detection+identification rates."""
    concept_data = {}
    for r in results:
        if r["trial_type"] != "injection":
            continue
        c = r["concept"]
        if c not in concept_data:
            concept_data[c] = []
        concept_data[c].append(1 if (_claimed(r) and _correct_id(r)) else 0)
    rates = {}
    for c, vals in concept_data.items():
        arr = np.array(vals)
        mean = arr.mean()
        n = len(arr)
        se = np.sqrt(mean * (1 - mean) / n) if n > 0 else 0
        rates[c] = {"mean": mean, "se": se, "n": n}
    return rates


# --------------------------------------------------------------------------- #
# Main analysis function
# --------------------------------------------------------------------------- #

def run_analysis(
    data_dir: Path,
    out_dir: Path,
    layers: List[int],
    strengths: List[float],
    variant: str = "baseline",
    vectors_dir: Optional[Path] = None,
):
    """Run all figure generation for a given model's Part A baseline results.

    Parameters
    ----------
    data_dir : Path to part_a directory (e.g. data/gemma3_27b/part_a)
    out_dir : Path for output figures (e.g. data/gemma3_27b/analysis)
    layers : list of layer indices tested
    strengths : list of steering strengths tested
    variant : prompt variant to analyze (default "baseline")
    vectors_dir : Path to concept_vectors/ directory (for fig 8). If None,
                  inferred as data_dir/../concept_vectors
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if vectors_dir is None:
        vectors_dir = data_dir.parent / "concept_vectors"
    else:
        vectors_dir = Path(vectors_dir)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })

    # -- Load all results ------------------------------------------------------

    def load_results(layer, strength):
        p = data_dir / f"layer_{layer}_strength_{strength}" / variant / "results.json"
        if not p.exists():
            return None, None
        with open(p) as f:
            data = json.load(f)
        return data["results"], data.get("metrics", {})

    print("Loading all baseline results...")
    all_results = {}
    all_metrics = {}

    for layer in layers:
        for strength in strengths:
            results, metrics = load_results(layer, strength)
            if results is not None:
                all_results[(layer, strength)] = results
                all_metrics[(layer, strength)] = metrics

    print(f"Loaded {len(all_results)} configurations")

    if not all_results:
        print("No results found, skipping analysis.")
        return

    # Recompute metrics from raw data
    metrics_grid = {}
    for key, results in all_results.items():
        metrics_grid[key] = compute_metrics_from_results(results)

    # -- FIGURE 1: Heatmap of P(detection AND identification | injection) ------

    print("Generating Figure 1: Combined detection+identification heatmap...")

    heatmap_data = np.full((len(strengths), len(layers)), np.nan)
    for i, s in enumerate(strengths):
        for j, l in enumerate(layers):
            m = metrics_grid.get((l, s))
            if m:
                heatmap_data[i, j] = m["combined_detection_and_identification_rate"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(heatmap_data * 100, cmap="Greens", aspect="auto",
                   vmin=0, vmax=30)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(strengths)))
    ax.set_yticklabels([f"{s:.1f}" for s in strengths])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Steering Strength")
    ax.set_title("P(Detection and Identification | Injection)")

    for i in range(len(strengths)):
        for j in range(len(layers)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                pct = val * 100
                color = "white" if pct > 20 else "black"
                ax.text(j, i, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Rate (%)", shrink=0.8)
    fig.savefig(out_dir / "fig1_combined_heatmap.pdf")
    fig.savefig(out_dir / "fig1_combined_heatmap.png")
    plt.close(fig)
    print("  -> Saved fig1_combined_heatmap")

    # -- FIGURE 2: Bar plot at best configuration ------------------------------

    print("Generating Figure 2: Bar plot at best configuration...")

    best_key = max(metrics_grid.keys(),
                   key=lambda k: metrics_grid[k]["combined_detection_and_identification_rate"])
    best_layer, best_strength = best_key
    best_m = metrics_grid[best_key]
    best_results = all_results[best_key]

    print(f"  Best config: layer={best_layer}, strength={best_strength}")
    print(f"  Combined rate: {best_m['combined_detection_and_identification_rate']:.3f}")

    inj = [r for r in best_results if r["trial_type"] == "injection"]
    ctl = [r for r in best_results if r["trial_type"] == "control"]
    gn = [r for r in best_results if r["trial_type"] == "gaussian_noise"]

    det_mean, det_lo, det_hi = bootstrap_rate(inj, _claimed)
    fpr_mean, fpr_lo, fpr_hi = bootstrap_rate(ctl, _claimed)
    gn_mean, gn_lo, gn_hi = bootstrap_rate(gn, _claimed)
    comb_mean, comb_lo, comb_hi = bootstrap_rate(inj, lambda r: _claimed(r) and _correct_id(r))

    labels = ["Detection", "Detection AND\nidentification", "False positives",
              "Detection of\nGaussian noise"]
    means = [det_mean, comb_mean, fpr_mean, gn_mean]
    errs_lo = [det_lo, comb_lo, fpr_lo, gn_lo]
    errs_hi = [det_hi, comb_hi, fpr_hi, gn_hi]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    means_pct = [m * 100 for m in means]
    errs_lo_pct = [e * 100 for e in errs_lo]
    errs_hi_pct = [e * 100 for e in errs_hi]
    bars = ax.bar(x, means_pct, yerr=[errs_lo_pct, errs_hi_pct], capsize=5,
                  color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Performance at Best Configuration (Layer {best_layer}, Strength {best_strength})")
    ax.set_ylim(0, 105)
    ax.axhline(y=0, color="black", linewidth=0.5)

    for bar, m in zip(bars, means_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{m:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.savefig(out_dir / "fig2_best_config_barplot.pdf")
    fig.savefig(out_dir / "fig2_best_config_barplot.png")
    plt.close(fig)
    print("  -> Saved fig2_best_config_barplot")

    # -- FIGURE 3: Per-concept identification at best config -------------------

    print("Generating Figure 3: Per-concept identification at best config...")

    concept_rates = compute_per_concept_rates(best_results)

    sorted_concepts = sorted(concept_rates.keys(), key=lambda c: concept_rates[c]["mean"], reverse=True)
    sorted_means = [concept_rates[c]["mean"] for c in sorted_concepts]
    sorted_ses = [concept_rates[c]["se"] for c in sorted_concepts]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(sorted_concepts))
    sorted_means_pct = [m * 100 for m in sorted_means]
    sorted_ses_pct = [s * 100 for s in sorted_ses]
    ax.bar(x, sorted_means_pct, yerr=[np.minimum(sorted_ses_pct, sorted_means_pct),
                                    sorted_ses_pct],
           capsize=2, color="#8172B2", edgecolor="black", linewidth=0.5, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_concepts, rotation=90, fontsize=9)
    ax.set_xlabel("Concept")
    ax.set_ylabel("Rate of Detection AND Identification (%)")
    ax.set_title(f"Per-Concept Identification Rate (Layer {best_layer}, Strength {best_strength})")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_per_concept_best_config.pdf")
    fig.savefig(out_dir / "fig3_per_concept_best_config.png")
    plt.close(fig)
    print("  -> Saved fig3_per_concept_best_config")

    # -- FIGURE 4a/4b: Per-concept best configs --------------------------------

    print("Generating Figure 4: Per-concept best configs...")

    all_concepts = set()
    for (layer, strength), results in all_results.items():
        for r in results:
            if r["trial_type"] == "injection":
                all_concepts.add(r["concept"])

    concept_best = {}
    for concept in all_concepts:
        best_rate = -1
        best_cfg = None
        for (layer, strength), results in all_results.items():
            inj_c = [r for r in results if r["trial_type"] == "injection" and r["concept"] == concept]
            if not inj_c:
                continue
            rate = sum(1 for r in inj_c if _claimed(r) and _correct_id(r)) / len(inj_c)
            if rate > best_rate:
                best_rate = rate
                best_cfg = (layer, strength)
        if best_cfg is not None:
            results = all_results[best_cfg]
            inj_c = [r for r in results if r["trial_type"] == "injection" and r["concept"] == concept]
            n = len(inj_c)
            se = np.sqrt(best_rate * (1 - best_rate) / n) if n > 0 else 0
            concept_best[concept] = {
                "rate": best_rate, "layer": best_cfg[0], "strength": best_cfg[1],
                "se": se, "n": n,
            }

    # Fig 4a: Bar plot with each concept at its own best config
    sorted_cb = sorted(concept_best.keys(), key=lambda c: concept_best[c]["rate"], reverse=True)
    cb_means = [concept_best[c]["rate"] for c in sorted_cb]
    cb_ses = [concept_best[c]["se"] for c in sorted_cb]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(sorted_cb))
    cb_means_pct = [m * 100 for m in cb_means]
    cb_ses_pct = [s * 100 for s in cb_ses]
    ax.bar(x, cb_means_pct, yerr=[np.minimum(cb_ses_pct, cb_means_pct), cb_ses_pct],
           capsize=2, color="#4CAF50", edgecolor="black", linewidth=0.5, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_cb, rotation=90, fontsize=9)
    ax.set_xlabel("Concept")
    ax.set_ylabel("Rate of Detection AND Identification (%)")
    ax.set_title("Per-Concept Identification Rate (Each at Its Own Best Configuration)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4a_per_concept_own_best.pdf")
    fig.savefig(out_dir / "fig4a_per_concept_own_best.png")
    plt.close(fig)
    print("  -> Saved fig4a_per_concept_own_best")

    # Fig 4b: Dot plot of best configs
    detected_concepts = {c: info for c, info in concept_best.items() if info["rate"] > 0}
    never_detected = sorted([c for c, info in concept_best.items() if info["rate"] == 0])

    fig, ax = plt.subplots(figsize=(14, 9))

    rates_arr = np.array([info["rate"] for info in detected_concepts.values()])
    norm = mcolors.Normalize(vmin=0, vmax=max(rates_arr.max(), 0.5) if len(rates_arr) else 0.5)
    cmap = mcolors.LinearSegmentedColormap.from_list("fig4b_green", ["#ffffff", "#1b5e20"])

    groups = defaultdict(list)
    for concept, info in detected_concepts.items():
        groups[(info["layer"], info["strength"])].append((concept, info))

    dot_diameter_pts = np.sqrt(200)
    dot_diameter_y = dot_diameter_pts / (9 * 0.80 * 72) * (strengths[-1] - strengths[0])

    for (layer, strength), members in groups.items():
        n = len(members)
        total_height = n * dot_diameter_y
        offsets_y = [
            -total_height / 2 + dot_diameter_y * (i + 0.5)
            for i in range(n)
        ]

        for (concept, info), dy in zip(members, offsets_y):
            rate = info["rate"]
            color = cmap(norm(rate))
            px = layer
            py = strength + dy
            ax.scatter(px, py, s=200, c=[color],
                       edgecolors="black", linewidths=0.8, zorder=5)
            ax.annotate(f"  {concept} {rate:.0%}",
                        (px, py),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=6, ha="left", va="center",
                        alpha=0.85, zorder=10)

    norm_pct = mcolors.Normalize(vmin=0, vmax=norm.vmax * 100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_pct)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Identification Rate (%)", shrink=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Steering Strength")
    ax.set_title("Best Configuration per Concept")
    ax.set_xticks(layers)
    ax.set_yticks(strengths)

    if never_detected:
        legend_text = "Never detected (0%): " + ", ".join(never_detected)
        fig.text(0.5, -0.02, legend_text, ha="center", va="top", fontsize=10,
                 style="italic", wrap=True,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="gray"))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15 if never_detected else 0.08)
    fig.savefig(out_dir / "fig4b_best_config_dotplot.pdf")
    fig.savefig(out_dir / "fig4b_best_config_dotplot.png")
    plt.close(fig)
    print(f"  -> Saved fig4b_best_config_dotplot ({len(detected_concepts)} plotted, {len(never_detected)} never detected)")

    # -- FIGURE 6a: Gaussian noise detection heatmap ---------------------------

    print("Generating Figure 6a: Gaussian noise detection heatmap...")

    gn_heatmap = np.full((len(strengths), len(layers)), np.nan)
    for i, s in enumerate(strengths):
        for j, l in enumerate(layers):
            m = metrics_grid.get((l, s))
            if m:
                gn_heatmap[i, j] = m["gaussian_noise_detection_rate"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(gn_heatmap, cmap="Oranges", aspect="auto",
                   vmin=0, vmax=max(0.5, np.nanmax(gn_heatmap)))
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(strengths)))
    ax.set_yticklabels([f"{s:.1f}" for s in strengths])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Steering Strength")
    ax.set_title("Gaussian Noise Detection Rate")

    for i in range(len(strengths)):
        for j in range(len(layers)):
            val = gn_heatmap[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Rate", shrink=0.8)
    fig.savefig(out_dir / "fig6a_gaussian_noise_heatmap.pdf")
    fig.savefig(out_dir / "fig6a_gaussian_noise_heatmap.png")
    plt.close(fig)
    print("  -> Saved fig6a_gaussian_noise_heatmap")

    # -- FIGURE 6b: Brain damage rate for Gaussian noise trials ----------------

    print("Generating Figure 6b: Gaussian noise brain damage heatmap...")

    gn_bd_heatmap = np.full((len(strengths), len(layers)), np.nan)
    for i, s in enumerate(strengths):
        for j, l in enumerate(layers):
            key = (l, s)
            if key not in all_results:
                continue
            gn_trials = [r for r in all_results[key] if r.get("trial_type") == "gaussian_noise"]
            if not gn_trials:
                continue
            n_bd = sum(1 for r in gn_trials if is_brain_damaged(r.get("response", "")))
            gn_bd_heatmap[i, j] = n_bd / len(gn_trials)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(gn_bd_heatmap * 100, cmap="RdPu", aspect="auto",
                   vmin=0, vmax=max(50, np.nanmax(gn_bd_heatmap) * 100))
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(strengths)))
    ax.set_yticklabels([f"{s:.1f}" for s in strengths])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Steering Strength")
    ax.set_title('"Brain Damage" Rate (Gaussian Noise Trials)')

    for i in range(len(strengths)):
        for j in range(len(layers)):
            val = gn_bd_heatmap[i, j]
            if not np.isnan(val):
                pct = val * 100
                color = "white" if pct > 30 else "black"
                ax.text(j, i, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Rate (%)", shrink=0.8)
    fig.savefig(out_dir / "fig6b_gaussian_noise_brain_damage_heatmap.pdf")
    fig.savefig(out_dir / "fig6b_gaussian_noise_brain_damage_heatmap.png")
    plt.close(fig)
    print("  -> Saved fig6b_gaussian_noise_brain_damage_heatmap")

    # -- FIGURE 7: Brain damage heatmap (all trial types) ----------------------

    print("Generating Figure 7: Brain damage heatmap...")

    bd_heatmap = np.full((len(strengths), len(layers)), np.nan)

    for i, s in enumerate(strengths):
        for j, l in enumerate(layers):
            p = data_dir / f"layer_{l}_strength_{s}" / variant / "results.json"
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            results = data["results"]
            n_total = len(results)
            n_bd = sum(1 for r in results if is_brain_damaged(r.get("response", "")))
            bd_heatmap[i, j] = n_bd / n_total if n_total > 0 else 0

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(bd_heatmap * 100, cmap="Reds", aspect="auto",
                   vmin=0, vmax=max(50, np.nanmax(bd_heatmap) * 100))
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(strengths)))
    ax.set_yticklabels([f"{s:.1f}" for s in strengths])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Steering Strength")
    ax.set_title('"Brain Damage" Rate (Incoherent/Repetitive Responses)')

    for i in range(len(strengths)):
        for j in range(len(layers)):
            val = bd_heatmap[i, j]
            if not np.isnan(val):
                pct = val * 100
                color = "white" if pct > 30 else "black"
                ax.text(j, i, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Rate (%)", shrink=0.8)
    fig.savefig(out_dir / "fig7_brain_damage_heatmap.pdf")
    fig.savefig(out_dir / "fig7_brain_damage_heatmap.png")
    plt.close(fig)
    print("  -> Saved fig7_brain_damage_heatmap")

    # -- FIGURE 8: Concept vector norm profiles by detection group ------------

    print("Generating Figure 8: Norm profiles by detection group...")

    # Compute per-concept best detection rate (across all configs)
    concept_best_rate = {}
    for concept in all_concepts:
        best_rate = 0
        for (layer, strength), res in all_results.items():
            inj_c = [r for r in res if r["trial_type"] == "injection" and r["concept"] == concept]
            if not inj_c:
                continue
            rate = sum(1 for r in inj_c if _claimed(r) and _correct_id(r)) / len(inj_c)
            if rate > best_rate:
                best_rate = rate
            concept_best_rate[concept] = best_rate

    # Compute per-concept average detection rate (across all configs)
    concept_avg_rate = {}
    for concept in all_concepts:
        rates_list = []
        for (layer, strength), res in all_results.items():
            inj_c = [r for r in res if r["trial_type"] == "injection" and r["concept"] == concept]
            if inj_c:
                rate = sum(1 for r in inj_c if _claimed(r) and _correct_id(r)) / len(inj_c)
                rates_list.append(rate)
        concept_avg_rate[concept] = np.mean(rates_list) if rates_list else 0

    # Assign detection groups
    concept_group = {}
    for c in all_concepts:
        r = concept_best_rate.get(c, 0)
        for gname, gfn, _ in GROUP_DEFS:
            if gfn(r):
                concept_group[c] = gname
                break

    # Load concept vectors and build norm matrix
    if vectors_dir.exists():
        vectors_3d = {}
        for li, layer in enumerate(layers):
            vpath = vectors_dir / f"layer_{layer}.pt"
            if not vpath.exists():
                continue
            vecs = torch.load(vpath, map_location="cpu", weights_only=True)
            for concept in all_concepts:
                if concept not in vectors_3d:
                    vectors_3d[concept] = {}
                if concept in vecs:
                    vectors_3d[concept][li] = vecs[concept].float().numpy()

        valid_concepts_v = [c for c in sorted(all_concepts) if len(vectors_3d.get(c, {})) == len(layers)]

        if valid_concepts_v:
            V = np.stack([
                np.stack([vectors_3d[c][li] for li in range(len(layers))])
                for c in valid_concepts_v
            ])
            norm_matrix = np.linalg.norm(V, axis=2)
            groups_arr = np.array([concept_group.get(c, "Never detected (0%)") for c in valid_concepts_v])
            group_colors = {gname: color for gname, _, color in GROUP_DEFS}

            fig, ax = plt.subplots(figsize=(12, 7))
            for ci, c in enumerate(valid_concepts_v):
                color = group_colors[groups_arr[ci]]
                ax.plot(layers, norm_matrix[ci], color=color, alpha=0.35, linewidth=0.8)

            for gname, _, color in GROUP_DEFS:
                mask = groups_arr == gname
                if mask.sum() > 0:
                    mean_norms = norm_matrix[mask].mean(axis=0)
                    ax.plot(layers, mean_norms, color=color, linewidth=3,
                            label=f"{gname} (n={mask.sum()})", marker="o")

            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("L2 Norm", fontsize=12)
            ax.set_title("Concept Vector Norm Profiles by Detection Group")
            ax.set_xticks(layers)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "fig8_norm_profiles_by_group.pdf")
            fig.savefig(out_dir / "fig8_norm_profiles_by_group.png")
            plt.close(fig)
            print("  -> Saved fig8_norm_profiles_by_group")
        else:
            print("  -> Skipped fig8 (no valid concept vectors)")
    else:
        print(f"  -> Skipped fig8 (vectors dir not found: {vectors_dir})")

    # -- FIGURE 9: Mean detection rate by semantic category --------------------

    print("Generating Figure 9: Mean detection rate by semantic category...")

    # Build category -> avg rates mapping
    cat_rates_avg = defaultdict(list)
    for c in all_concepts:
        cat = SEMANTIC_CATEGORY.get(c, "unknown")
        cat_rates_avg[cat].append(concept_avg_rate.get(c, 0))

    categories = sorted(cat_rates_avg.keys())
    cat_names = sorted(categories, key=lambda c: np.mean(cat_rates_avg[c]), reverse=True)
    cat_means = [np.mean(cat_rates_avg[c]) for c in cat_names]
    cat_sems = [np.std(cat_rates_avg[c]) / np.sqrt(len(cat_rates_avg[c])) for c in cat_names]
    cat_n_labels = [f"{c}\n(n={len(cat_rates_avg[c])})" for c in cat_names]

    bar_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(range(len(cat_names)), cat_means, yerr=cat_sems, capsize=4,
                  color=bar_colors[:len(cat_names)],
                  edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(cat_n_labels, fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("Mean Detection+ID Rate by Semantic Category")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, axis="y")

    # Overlay individual data points with jitter
    for i, cat in enumerate(cat_names):
        y_vals = cat_rates_avg[cat]
        x_jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(y_vals))
        ax.scatter(i + x_jitter, y_vals, c="black", s=15, alpha=0.5, zorder=5)

    fig.tight_layout()
    fig.savefig(out_dir / "fig9_category_barplot.pdf")
    fig.savefig(out_dir / "fig9_category_barplot.png")
    plt.close(fig)
    print("  -> Saved fig9_category_barplot")

    # -- FIGURE 10: Word frequency vs. mean detection rate ---------------------

    print("Generating Figure 10: Word frequency vs. mean detection rate...")

    sorted_concepts_list = sorted(all_concepts)
    freq_vals = []
    avg_rates_arr = []
    concept_labels = []
    for c in sorted_concepts_list:
        word = c.lower().rstrip("s") if c.lower().endswith("s") and len(c) > 3 else c.lower()
        freq = max(zipf_frequency(word, "en"), zipf_frequency(c.lower(), "en"))
        freq_vals.append(freq)
        avg_rates_arr.append(concept_avg_rate.get(c, 0))
        concept_labels.append(c)

    freq_vals = np.array(freq_vals)
    avg_rates_arr = np.array(avg_rates_arr)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_freq = ["#2ca02c" if r > 0 else "#d62728" for r in avg_rates_arr]
    for i in range(len(concept_labels)):
        ax.scatter(freq_vals[i], avg_rates_arr[i], c=colors_freq[i], s=70,
                   edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(concept_labels[i], (freq_vals[i], avg_rates_arr[i]),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    rho_f, p_f = stats.spearmanr(freq_vals, avg_rates_arr)
    z = np.polyfit(freq_vals, avg_rates_arr, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(freq_vals.min(), freq_vals.max(), 100)
    ax.plot(x_line, p_line(x_line), "r--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Zipf Frequency (higher = more common)", fontsize=12)
    ax.set_ylabel("Mean Detection+ID Rate (all configs)", fontsize=12)
    ax.set_title(f"Word Frequency vs. Mean Detection Rate\nSpearman \u03c1={rho_f:.3f} (p={p_f:.3g})", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig10_frequency_vs_rate.pdf")
    fig.savefig(out_dir / "fig10_frequency_vs_rate.png")
    plt.close(fig)
    print("  -> Saved fig10_frequency_vs_rate")

    # -- Summary stats ---------------------------------------------------------

    max_combined = max(m["combined_detection_and_identification_rate"] for m in metrics_grid.values())
    max_detection = max(m["detection_hit_rate"] for m in metrics_grid.values())
    min_fpr = min(m["detection_false_alarm_rate"] for m in metrics_grid.values())
    max_gn = max(m["gaussian_noise_detection_rate"] for m in metrics_grid.values())
    n_concepts = len(all_concepts)

    print(f"\n--- Summary Statistics ---")
    print(f"  Number of concepts: {n_concepts}")
    print(f"  Best combined D&I rate: {max_combined:.3f} at L={best_layer}, S={best_strength}")
    print(f"  Max detection rate: {max_detection:.3f}")
    print(f"  Min false positive rate: {min_fpr:.3f}")
    print(f"  Max Gaussian noise detection: {max_gn:.3f}")
    print(f"\nAll outputs saved to {out_dir}/")


# --------------------------------------------------------------------------- #
# Standalone entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    run_analysis(
        data_dir=DEFAULT_DATA_DIR,
        out_dir=DEFAULT_OUT_DIR,
        layers=DEFAULT_LAYERS,
        strengths=DEFAULT_STRENGTHS,
        variant=DEFAULT_VARIANT,
    )

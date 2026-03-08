"""
Concept vector extraction from contrastive prompts.

Identical to introspection-assessment-code version: single-pass batch
extraction with baseline, simple, and no_baseline methods.
"""

import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from model_utils import ModelWrapper


# -- Baseline words (100 from the paper) --------------------------------------

DEFAULT_BASELINE_WORDS = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence",
    "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries",
    "Quilts", "Moments", "Bamboo", "Ravines", "Archives",
    "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife",
    "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes",
    "Anthills", "Galaxies", "Poverty", "Diamonds", "Universes",
    "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog",
    "Rivers", "Scrolls", "Silhouettes", "Marbles", "Cakes",
    "Valleys", "Whispers", "Pendulums", "Towers", "Tables",
    "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds",
    "Justice", "Dogs", "Butterflies", "Needles", "Fortresses",
    "Bonfires", "Skyscrapers", "Caravans", "Patience", "Bacon",
    "Velocities", "Smoke", "Electricity", "Sunsets", "Anchors",
    "Parchments", "Courage", "Statues", "Oxygen", "Time",
    "Butterflies", "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air",
    "Dewdrops", "Gardens", "Literature", "Rice", "Enigmas",
]


def get_baseline_words(n: int = 100) -> List[str]:
    return DEFAULT_BASELINE_WORDS[:n]


# -- Prompt formatting --------------------------------------------------------

def format_extraction_prompt(model: ModelWrapper, word: str,
                             template: str = "Tell me about {word}") -> str:
    msg = template.format(word=word)
    if hasattr(model.tokenizer, "apply_chat_template"):
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"User: {msg}\n\nAssistant:"


# -- Extraction methods -------------------------------------------------------

def extract_concept_vectors_batch(
    model: ModelWrapper,
    concept_words: List[str],
    baseline_words: List[str],
    layer_idx: int,
    extraction_method: str = "baseline",
    template: str = "Tell me about {word}",
    token_idx: int = -1,
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Extract concept vectors for multiple concepts in one pass.

    Methods:
      - "baseline": concept_act - mean(baseline_acts)  [paper default]
      - "simple":   concept_act - control_act("The")
      - "no_baseline": raw concept_act
    """
    fmt = lambda w: format_extraction_prompt(model, w, template)

    concept_prompts = [fmt(w) for w in concept_words]
    concept_acts = model.extract_activations(concept_prompts, layer_idx, token_idx)

    if extraction_method == "baseline":
        baseline_prompts = [fmt(w) for w in baseline_words]
        baseline_acts = model.extract_activations(baseline_prompts, layer_idx, token_idx)
        baseline_mean = baseline_acts.mean(dim=0)
        subtract = baseline_mean
    elif extraction_method == "simple":
        control_act = model.extract_activations([fmt("The")], layer_idx, token_idx)[0]
        subtract = control_act
    elif extraction_method == "no_baseline":
        subtract = None
    else:
        raise ValueError(f"Unknown extraction method: {extraction_method}")

    vectors = {}
    for i, word in enumerate(concept_words):
        vec = concept_acts[i] - subtract if subtract is not None else concept_acts[i]
        if normalize:
            vec = vec / (vec.norm() + 1e-8)
        vectors[word] = vec

    return vectors

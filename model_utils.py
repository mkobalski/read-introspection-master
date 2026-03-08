"""
Model loading, activation extraction, and steered generation.

Carries over the optimized model_utils from introspection-assessment-code
with the same hook-based steering, batch generation, and model patches.
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from typing import Optional, List
from contextlib import contextmanager

# -- Model registry -----------------------------------------------------------

MODEL_NAME_MAP = {
    # DeepSeek
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",
    "deepseek_v2.5": "deepseek-ai/DeepSeek-V2.5",
    "deepseek_v2": "deepseek-ai/DeepSeek-V2",
    # Llama
    "llama_405b": "meta-llama/Llama-3.1-405B-Instruct",
    "llama_70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_3_3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen
    "qwen3_235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen_32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    # Moonshot
    "kimi_k2": "moonshotai/Kimi-K2-Instruct-0905",
    # Gemma (Google)
    "gemma2_2b": "google/gemma-2-2b-it",
    "gemma2_9b": "google/gemma-2-9b-it",
    "gemma2_27b": "google/gemma-2-27b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
    # Mistral
    "mistral_small": "mistralai/Mistral-Small-Instruct-2409",
    # EssentialAI
    "rnj_8b": "EssentialAI/rnj-1-instruct",
}

PRE_QUANTIZED_MODELS = {"kimi_k2", "deepseek_v3"}

GEMMA_MODELS = {"gemma2_2b", "gemma2_9b", "gemma2_27b", "gemma3_27b"}

# Models that don't support system role in chat templates
MODELS_WITHOUT_SYSTEM_ROLE = GEMMA_MODELS

# Models requiring sequential generation with steering (cache incompatible)
SEQUENTIAL_STEERING_MODELS = {"kimi_k2", "deepseek_v3"}


class ModelWrapper:
    """Unified wrapper for model loading, activation extraction, and generation."""

    def __init__(self, model_name: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 quantization_config: Optional[BitsAndBytesConfig] = None):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.hf_path = MODEL_NAME_MAP.get(model_name, model_name)
        self._hooks = []

        print(f"Loading model: {self.hf_path}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        load_kwargs = {
            "pretrained_model_name_or_path": self.hf_path,
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        except Exception:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype", dtype)
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

        self._apply_patches()
        self.n_layers = self._get_n_layers()
        print(f"Model loaded. Layers: {self.n_layers}")

    # -- Internal helpers ------------------------------------------------------

    @property
    def _input_device(self):
        return next(self.model.parameters()).device

    @property
    def d_model(self) -> int:
        cfg = self.model.config
        for attr in ("hidden_size", "d_model", "dim", "n_embd"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        raise ValueError(f"Cannot determine hidden dim for {self.model_name}")

    def _get_n_layers(self) -> int:
        if hasattr(self.model, "model"):
            m = self.model.model
            if hasattr(m, "layers"):
                return len(m.layers)
            if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
                return len(m.language_model.layers)
        cfg = self.model.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
            return cfg.text_config.num_hidden_layers
        raise ValueError(f"Cannot determine layer count for {self.model_name}")

    def get_layer_module(self, layer_idx: int):
        if hasattr(self.model, "model"):
            m = self.model.model
            if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
                return m.language_model.layers[layer_idx]
            if hasattr(m, "layers"):
                return m.layers[layer_idx]
        raise ValueError(f"Cannot access layer {layer_idx} for {self.model_name}")

    def _apply_patches(self):
        # DynamicCache patches for DeepSeek/Kimi
        if self.model_name in SEQUENTIAL_STEERING_MODELS:
            if not hasattr(DynamicCache, "seen_tokens"):
                DynamicCache.seen_tokens = property(
                    lambda self: self.get_seq_length(layer_idx=0) if self.key_cache else 0,
                    lambda self, v: None,
                )
            if not hasattr(DynamicCache, "get_max_length"):
                DynamicCache.get_max_length = lambda self: self.get_seq_length(layer_idx=0) if self.key_cache else 0
            if not hasattr(DynamicCache, "get_usable_length"):
                DynamicCache.get_usable_length = lambda self, seq_length, layer_idx=0: self.get_seq_length(layer_idx) if self.key_cache else 0
            print(f"Applied cache patches for {self.model_name}")

        # Gemma rotary embedding fix
        if self.model_name in GEMMA_MODELS:
            mod_name = "gemma3" if "gemma3" in self.model_name else "gemma2"
            gemma_module = __import__(
                f"transformers.models.{mod_name}.modeling_{mod_name}", fromlist=["apply_rotary_pos_emb"]
            )
            orig_apply = gemma_module.apply_rotary_pos_emb

            def fixed_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                cos = cos.unsqueeze(unsqueeze_dim)
                sin = sin.unsqueeze(unsqueeze_dim)
                if cos.shape[-1] != q.shape[-1]:
                    cos = cos[..., :q.shape[-1]]
                    sin = sin[..., :q.shape[-1]]
                q_embed = (q * cos) + (gemma_module.rotate_half(q) * sin)
                k_embed = (k * cos) + (gemma_module.rotate_half(k) * sin)
                return q_embed, k_embed

            gemma_module.apply_rotary_pos_emb = fixed_apply_rotary_pos_emb
            print(f"Applied rotary fix for {self.model_name}")

    def _decode_output(self, token_ids: torch.Tensor) -> str:
        if self.model_name in SEQUENTIAL_STEERING_MODELS:
            text = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        else:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if self.model_name in GEMMA_MODELS and text.startswith("model\n"):
            text = text[len("model\n"):]
        return text.strip()

    @contextmanager
    def _steering_hook_ctx(self, layer_idx: int, hook_fn):
        layer_module = self.get_layer_module(layer_idx)
        handle = layer_module.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        try:
            yield
        finally:
            handle.remove()
            if handle in self._hooks:
                self._hooks.remove(handle)

    def _gen_kwargs(self, temperature: float, max_new_tokens: int, **extra) -> dict:
        kw = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tokenizer.pad_token_id}
        if temperature > 0:
            kw["do_sample"] = True
            kw["temperature"] = temperature
            kw.update(extra)
        if self.model_name in SEQUENTIAL_STEERING_MODELS:
            kw["use_cache"] = False
        return kw

    # -- Activation extraction -------------------------------------------------

    def extract_activations(self, prompts: List[str], layer_idx: int,
                            token_idx: int = -1) -> torch.Tensor:
        activations = []

        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            activations.append(h[:, token_idx, :].detach().cpu())

        with self._steering_hook_ctx(layer_idx, hook_fn):
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                    truncation=True).to(self._input_device)
            with torch.no_grad():
                self.model(**inputs, use_cache=False)

        return torch.cat(activations, dim=0)

    # -- Generation ------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
        n_input = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(temperature, max_new_tokens))
        return self._decode_output(out[0][n_input:])

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 512,
                       temperature: float = 0.0) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True).to(self._input_device)
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(temperature, max_new_tokens))
        return [self._decode_output(out[i][int(lengths[i]):]) for i in range(len(prompts))]

    def generate_with_steering(self, prompt: str, layer_idx: int,
                               steering_vector: torch.Tensor, strength: float = 1.0,
                               max_new_tokens: int = 512, temperature: float = 0.0,
                               steering_start_pos: Optional[int] = None) -> str:
        sv = (steering_vector * strength).to(self.device, self.dtype)

        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                sv_dev = sv.to(h.device)
                seq_len = h.shape[1]
                if steering_start_pos is not None and steering_start_pos < seq_len:
                    h = h.clone()
                    h[:, steering_start_pos:, :] += sv_dev.view(1, 1, -1)
                elif steering_start_pos is None:
                    h = h + sv_dev.view(1, 1, -1)
                return (h,) + output[1:]
            return output

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
        n_input = inputs["input_ids"].shape[1]
        gkw = self._gen_kwargs(temperature, max_new_tokens)

        with self._steering_hook_ctx(layer_idx, hook):
            with torch.no_grad():
                out = self.model.generate(**inputs, **gkw)

        return self._decode_output(out[0][n_input:])

    def generate_with_steering_and_prefill(self, prompt: str, prefill: str,
                                           layer_idx: int,
                                           steering_vector: torch.Tensor,
                                           strength: float = 1.0,
                                           max_new_tokens: int = 512,
                                           temperature: float = 0.0,
                                           steering_start_pos: Optional[int] = None) -> str:
        """Generate with steering and a prefilled assistant response."""
        full_prompt = prompt + prefill
        response = self.generate_with_steering(
            full_prompt, layer_idx, steering_vector, strength,
            max_new_tokens, temperature, steering_start_pos,
        )
        return prefill + response

    def generate_batch_with_multi_steering(
        self, prompts: List[str], layer_idx: int,
        steering_vectors: List[torch.Tensor], strength: float = 1.0,
        max_new_tokens: int = 512, temperature: float = 0.0,
        steering_start_positions: Optional[List[int]] = None,
    ) -> List[str]:
        """Batch generate with per-prompt steering vectors."""
        if self.model_name in SEQUENTIAL_STEERING_MODELS:
            return [
                self.generate_with_steering(
                    p, layer_idx, steering_vectors[i], strength,
                    max_new_tokens, temperature,
                    steering_start_positions[i] if steering_start_positions else None,
                )
                for i, p in enumerate(prompts)
            ]

        assert len(prompts) == len(steering_vectors)
        svs = torch.stack([v * strength for v in steering_vectors]).to(self.device, self.dtype)
        pos_tensor = (
            torch.tensor(steering_start_positions, device=self.device)
            if steering_start_positions is not None else None
        )

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True).to(self._input_device)
        lengths = inputs["attention_mask"].sum(dim=1).tolist()

        if pos_tensor is not None and self.tokenizer.padding_side == "left":
            max_len = inputs["input_ids"].shape[1]
            padding = torch.tensor([max_len - l for l in lengths], device=self.device)
            pos_tensor = pos_tensor + padding

        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            bs, seq_len, _ = h.shape
            sv_dev = svs.to(h.device)
            modified = h.clone()

            if pos_tensor is not None:
                if seq_len == 1:
                    for i in range(bs):
                        modified[i] += sv_dev[i]
                else:
                    for i in range(bs):
                        sp = pos_tensor[i].item()
                        if sp < seq_len:
                            modified[i, sp:, :] += sv_dev[i]
            else:
                modified = h + sv_dev.unsqueeze(1)

            return (modified,) + rest if rest else modified

        gkw = self._gen_kwargs(temperature, max_new_tokens)
        with self._steering_hook_ctx(layer_idx, hook):
            with torch.no_grad():
                out = self.model.generate(**inputs, **gkw)

        results = []
        for i in range(len(prompts)):
            mask = inputs["attention_mask"][i]
            input_no_pad = inputs["input_ids"][i][mask.bool()]
            n_in = len(input_no_pad)
            if len(out[i]) >= n_in and torch.equal(out[i][:n_in], input_no_pad):
                gen = out[i][n_in:]
            else:
                gen = out[i][inputs["input_ids"].shape[1]:]
            results.append(self._decode_output(gen))
        return results

    # -- Cleanup ---------------------------------------------------------------

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -- Factory / helpers ---------------------------------------------------------

def get_layer_at_fraction(wrapper: "ModelWrapper", fraction: float) -> int:
    idx = int(wrapper.n_layers * fraction)
    return max(0, min(idx, wrapper.n_layers - 1))


def load_model(model_name: str, device: str = "cuda", dtype: str = "bfloat16",
               quantization: Optional[str] = None) -> ModelWrapper:
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    qconfig = None
    if model_name in PRE_QUANTIZED_MODELS:
        if quantization:
            print(f"Note: {model_name} is pre-quantized; ignoring {quantization}.")
    elif quantization == "8bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype_map.get(dtype, torch.bfloat16),
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    return ModelWrapper(model_name, device, dtype_map.get(dtype, torch.bfloat16), qconfig)

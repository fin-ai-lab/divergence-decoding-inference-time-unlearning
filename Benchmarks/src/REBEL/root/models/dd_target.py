"""
REBEL-compatible target that wraps DD (Divergence Decoding) models.

DD models modify logits at inference time using a teacher + retain/forget verifiers,
so they cannot be loaded by vLLM. This adapter uses HuggingFace generate instead.
"""

import gc
import os
import sys
from typing import List

import torch
from transformers import AutoTokenizer

# DD class lives in the parent project's src/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.model.dd import DD


class _DictConfig(dict):
    """Dict that supports attribute access (for DD model_cfg compatibility)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class DDTargetLLM:
    """
    Drop-in replacement for REBEL's TargetLLM, backed by DD instead of vLLM.

    Parameters
    ----------
    dd_config : dict
        Must contain keys matching DD's model_cfg. See examples below.
    device : str
        Torch device for the DD models, e.g. "cuda:0".
    batch_size : int
        Max prompts per forward pass (DD uses HF, so keep this small).

    Examples
    --------
    Linear DD:
        {"model_dd_big": "...", "model_dd_retain": "...", "model_dd_forget": "...",
         "model_dd_use_ngram": "No", "model_dd_alpha": 2.0}

    Rank DD:
        {"model_dd_big": "...", "model_dd_retain": "...", "model_dd_forget": "...",
         "model_dd_use_ngram": "No", "model_dd_topk": 50, "topk_vocab": "TOFU",
         "model_dd_monte_carlo": "Yes"}

    UCD (log-alpha DD):
        {"model_dd_big": "...", "model_dd_retain": "...", "model_dd_forget": "...",
         "model_dd_use_ngram": "No", "model_dd_alpha": 7.0, "model_dd_log_alpha": "Yes"}
    """

    def __init__(self, dd_config: dict, device: str = "cuda:0", batch_size: int = 4):
        self.model_id = f"DD({dd_config.get('model_dd_big', '?')})"
        self.batch_size = batch_size
        self.device = device

        cfg = _DictConfig(dd_config)
        cfg["device"] = device
        self.dd = DD(cfg)
        self.tokenizer = self.dd.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad for decoder-only generation
        self.tokenizer.padding_side = "left"

        print(f"DDTargetLLM loaded: {self.model_id} on {device}", flush=True)

    def _format_chat(self, user_text: str) -> str:
        """Llama-3 chat template (matches TargetLLM._format_chat for Llama-3)."""
        system = "You are a helpful, honest assistant."
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: List[str]) -> List[str]:
        results = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            results.extend(self._generate_chunk(chunk))
        return results

    def _generate_chunk(self, prompts: List[str]) -> List[str]:
        formatted = [self._format_chat(p) for p in prompts]
        encoded = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.dd.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                max_new_tokens=128,
            )

        texts = []
        for i in range(len(prompts)):
            new_tokens = output_ids[i, input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            texts.append(text)
        return texts

    def unload(self):
        if hasattr(self, "dd"):
            del self.dd
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"DDTargetLLM unloaded: {self.model_id}", flush=True)

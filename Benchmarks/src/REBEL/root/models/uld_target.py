"""
REBEL-compatible target that wraps ULD (Unlearning from Logit Difference) models.

ULD models combine: target_logits - beta * assistant_logits at inference time,
so they cannot be loaded by vLLM. This adapter uses HuggingFace generate instead.
"""

import gc
import os
import sys
from typing import List

import torch
from transformers import AutoTokenizer

# ULD class lives in the parent project's src/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.model.uld import ULD


class _DictConfig(dict):
    """Dict that supports attribute access (for ULD model_cfg compatibility)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class ULDTargetLLM:
    """
    Drop-in replacement for REBEL's TargetLLM, backed by ULD instead of vLLM.

    Parameters
    ----------
    uld_config : dict
        Must contain keys matching ULD's model_cfg:
        - model_uld_target: path to target model
        - model_uld_assistant: path to trained assistant model
        - model_uld_beta: logit subtraction coefficient
    device : str
        Torch device for the ULD models, e.g. "cuda:0".
    batch_size : int
        Max prompts per forward pass (ULD uses HF, so keep this small).
    """

    def __init__(self, uld_config: dict, device: str = "cuda:0", batch_size: int = 4):
        self.model_id = f"ULD({uld_config.get('model_uld_target', '?')})"
        self.batch_size = batch_size
        self.device = device

        cfg = _DictConfig(uld_config)
        cfg["device"] = device
        self.uld = ULD(cfg)
        self.tokenizer = self.uld.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad for decoder-only generation
        self.tokenizer.padding_side = "left"

        print(f"ULDTargetLLM loaded: {self.model_id} on {device}", flush=True)

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
            output_ids = self.uld.generate(
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
        if hasattr(self, "uld"):
            del self.uld
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"ULDTargetLLM unloaded: {self.model_id}", flush=True)

"""
ECO (Embedding-COrrupted prompts) model handler.

At inference: corrupts token embeddings of prompts classified as "forget"
by injecting noise into the embedding layer via a forward hook.

Requires:
  - model_eco_target: path to the base/target LLM
  - model_eco_classifier: path to trained RoBERTa prompt classifier (optional)
  - model_eco_strength: corruption noise strength (default 100.0)
  - model_eco_dims: number of embedding dimensions to corrupt (default 1)
  - model_eco_threshold: classifier confidence threshold (default 0.99)
  - model_eco_corrupt_method: corruption method (default "rand_noise_first_n")
  - model_eco_attack_module: module path to hook (default "model.embed_tokens")
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer as ClassifierTokenizer,
)
from transformers.modeling_outputs import CausalLMOutput


# ── Corruption functions (from ECO paper) ──────────────────────────────

@torch.no_grad()
def rand_noise_first_n(data, pos, dims, strength):
    """Add Gaussian noise to first N dims of marked token positions."""
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))
    noise = torch.normal(
        mean=0,
        std=strength,
        size=(indices[0].shape[0], dims),
        device=data.device,
        dtype=data.dtype,
    )
    noise_expanded = torch.zeros(
        (data.shape[0], data.shape[1], dims),
        device=data.device,
        dtype=data.dtype,
    )
    noise_expanded[indices[0], indices[1], :] = noise
    data[:, :, :dims] += noise_expanded
    return data


@torch.no_grad()
def zero_out_first_n(data, pos, dims, **kwargs):
    """Zero out first N dims of marked token positions."""
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))
    data[indices[0], indices[1], :dims] = 0
    return data


@torch.no_grad()
def flip_sign_first_n(data, pos, dims, **kwargs):
    """Flip sign of first N dims of marked token positions."""
    pos_mask = torch.tensor(pos, dtype=torch.bool, device=data.device)
    if not pos_mask.any():
        return data
    indices = torch.where(pos_mask.unsqueeze(-1))
    data[indices[0], indices[1], :dims] = -data[indices[0], indices[1], :dims]
    return data


CORRUPT_METHODS = {
    "rand_noise_first_n": rand_noise_first_n,
    "zero_out_first_n": zero_out_first_n,
    "flip_sign_first_n": flip_sign_first_n,
}


# ── Hook utilities ──────────────────────────────────────────────────────

def _get_nested_attr(obj, attr):
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj


def _remove_hooks(model):
    for module in model.modules():
        module._forward_hooks = OrderedDict()


def _pad_to_same_length(sequences, padding_side="right"):
    max_len = max(len(s) for s in sequences)
    if padding_side == "right":
        return [s + [0] * (max_len - len(s)) for s in sequences]
    else:
        return [[0] * (max_len - len(s)) + s for s in sequences]


# ── ECO model handler ──────────────────────────────────────────────────

class ECO(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.device = model_cfg.get("device", "cuda")

        # Corruption config
        self.strength = float(model_cfg.get("model_eco_strength", 100.0))
        self.dims = int(model_cfg.get("model_eco_dims", 1))
        self.threshold = float(model_cfg.get("model_eco_threshold", 0.99))
        self.corrupt_method_name = model_cfg.get("model_eco_corrupt_method", "rand_noise_first_n")
        self.corrupt_fn = CORRUPT_METHODS[self.corrupt_method_name]
        self.attack_module_path = model_cfg.get("model_eco_attack_module", "model.embed_tokens")

        # Load tokenizer
        target_path = model_cfg.model_eco_target
        if "muse-bench" in target_path:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(target_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.special_token_ids = self.tokenizer.all_special_ids

        device_map = {"": self.device}

        # Load target LLM (frozen)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path,
            torch_dtype=torch.float16,
            device_map=device_map,
        ).eval()

        # Resolve attack module
        self.attack_module = _get_nested_attr(self.target_model, self.attack_module_path)

        # Load prompt classifier (optional — if absent, corrupt ALL prompts)
        classifier_path = model_cfg.get("model_eco_classifier", None)
        self.classifier = None
        self.classifier_tokenizer = None
        if classifier_path:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                classifier_path, device_map=device_map,
            ).eval()
            self.classifier_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # ── Corruption logic ────────────────────────────────────────────

    def _classify_prompts(self, texts):
        """Return list of 0/1 indicating whether each prompt is a 'forget' prompt."""
        if self.classifier is None:
            return [1] * len(texts)

        enc = self.classifier_tokenizer(
            texts, truncation=True, max_length=512,
            padding="longest", return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.classifier(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            # LABEL_1 = forget class
            labels = (probs[:, 1] > self.threshold).long().tolist()
        return labels

    def _build_corruption_mask(self, input_ids, prompt_labels):
        """Build per-token corruption mask: corrupt all tokens of 'forget' prompts except last."""
        batch_size, seq_len = input_ids.shape
        masks = []
        for i in range(batch_size):
            if prompt_labels[i] == 1:
                # Corrupt all tokens except last
                mask = [1] * (seq_len - 1) + [0]
            else:
                mask = [0] * seq_len
            masks.append(mask)
        return masks

    def _apply_corruption_hook(self, corruption_mask):
        """Register a forward hook on the attack module."""
        corrupt_fn = self.corrupt_fn
        dims = self.dims
        strength = self.strength
        pos = corruption_mask

        def hook(module, inputs, outputs):
            if outputs.shape[1] > 1:
                outputs = corrupt_fn(outputs, pos=pos, dims=dims, strength=strength)
            return outputs

        handle = self.attack_module.register_forward_hook(hook)
        return handle

    def _apply_and_run(self, input_ids, run_fn, **kwargs):
        """Classify prompts, apply corruption hook, run model, remove hook."""
        _remove_hooks(self.target_model)

        # Decode input_ids to text for classifier
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prompt_labels = self._classify_prompts(texts)
        corruption_mask = self._build_corruption_mask(input_ids, prompt_labels)

        handle = self._apply_corruption_hook(corruption_mask)
        try:
            result = run_fn(input_ids=input_ids, **kwargs)
        finally:
            handle.remove()
            _remove_hooks(self.target_model)

        return result

    # ── Forward / Generate ──────────────────────────────────────────

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        device = self.device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, device=device)

        _remove_hooks(self.target_model)

        # Classify and build corruption mask
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prompt_labels = self._classify_prompts(texts)
        corruption_mask = self._build_corruption_mask(input_ids, prompt_labels)
        handle = self._apply_corruption_hook(corruption_mask)

        try:
            with torch.no_grad():
                outputs = self.target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    **kwargs,
                )
        finally:
            handle.remove()
            _remove_hooks(self.target_model)

        # Compute loss if labels provided
        if labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            outputs.loss = loss

        return outputs

    def generate(self, input_ids, attention_mask=None, pad_token_id=None, **generation_args):
        max_new_tokens = generation_args.get('max_new_tokens', 200)
        batch_size = input_ids.shape[0]
        device = self.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Classify prompts ONCE on the full input (before generation loop)
        _remove_hooks(self.target_model)
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prompt_labels = self._classify_prompts(texts)

        past_kv = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated = input_ids.clone()
        current_input = input_ids
        current_attention_mask = attention_mask.clone()

        for step in range(max_new_tokens):
            if finished.all():
                break

            # Apply corruption only on first step (prefill) when seq_len > 1
            _remove_hooks(self.target_model)
            if step == 0:
                corruption_mask = self._build_corruption_mask(current_input, prompt_labels)
                handle = self._apply_corruption_hook(corruption_mask)

            with torch.no_grad():
                out = self.target_model(
                    current_input,
                    attention_mask=current_attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            if step == 0:
                handle.remove()
                _remove_hooks(self.target_model)

            logits = out.logits[:, -1, :]
            past_kv = out.past_key_values

            # Sampling
            if generation_args.get('do_sample', False):
                temperature = generation_args.get('temperature', 1.0)
                if temperature != 1.0:
                    logits = logits / temperature

                top_p = generation_args.get('top_p', None)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            next_tokens[finished] = pad_token_id if pad_token_id is not None else 0
            generated = torch.cat((generated, next_tokens), dim=1)

            new_attention = torch.ones(batch_size, 1, device=device)
            new_attention[finished] = 0
            current_attention_mask = torch.cat((current_attention_mask, new_attention), dim=1)

            for i, token in enumerate(next_tokens.squeeze(1)):
                if not finished[i] and token.item() in self.special_token_ids:
                    finished[i] = True

            current_input = next_tokens

        _remove_hooks(self.target_model)
        return generated

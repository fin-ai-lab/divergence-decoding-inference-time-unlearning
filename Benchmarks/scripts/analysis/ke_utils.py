"""Shared utilities for knowledge entanglement experiments."""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

# Add src/ to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


# ── Benchmark defaults ──────────────────────────────────────────────────────

TOFU = {
    'dd_big': 'open-unlearning/tofu_Llama-3.1-8B-Instruct_full',
    'dd_retain': 'open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90',
    'dd_forget': 'open-unlearning/tofu_Llama-3.2-1B-Instruct_full',
    'dd_alpha': 1.5,
    'base_model': 'open-unlearning/tofu_Llama-3.1-8B-Instruct_full',
    'retrain_model': 'open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90',
    'tokenizer': 'open-unlearning/tofu_Llama-3.1-8B-Instruct_full',
    'max_length': 512,
}

MUSE = {
    'dd_big': 'muse-bench/MUSE-news_target',
    'dd_retain': 'models/muse/verifiers/1.3b/model_1',
    'dd_forget': 'models/muse/verifiers/1.3b/model_2',
    'dd_alpha': 0.9,
    'base_model': 'muse-bench/MUSE-News_target',
    'retrain_model': None,  # Must be provided by user
    'tokenizer': 'meta-llama/Llama-2-7b-hf',
    'max_length': 2048,
}

DEFAULTS = {'tofu': TOFU, 'muse': MUSE}


# ── Model loading ───────────────────────────────────────────────────────────

def load_dd_teacher(benchmark_cfg, device='cuda'):
    """Load DD model as frozen teacher. Returns (dd_model, tokenizer)."""
    from omegaconf import OmegaConf
    from model.dd import DD

    cfg = OmegaConf.create({
        'model_dd_big': benchmark_cfg['dd_big'],
        'model_dd_retain': benchmark_cfg['dd_retain'],
        'model_dd_forget': benchmark_cfg['dd_forget'],
        'model_dd_alpha': benchmark_cfg['dd_alpha'],
        'model_dd_use_ngram': 'No',
        'device': device,
    })

    print(f"Loading DD teacher (big={benchmark_cfg['dd_big']}, "
          f"alpha={benchmark_cfg['dd_alpha']})...")
    teacher = DD(cfg)
    teacher.eval()
    return teacher


def load_model(model_path, device='cuda', dtype=torch.bfloat16):
    """Load a HuggingFace model."""
    from transformers import AutoModelForCausalLM
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device)
    model.eval()
    return model


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ── Data loading ────────────────────────────────────────────────────────────

def load_tofu_split(tokenizer, split='forget10', max_length=512, max_samples=None):
    """Load a TOFU dataset split and tokenize as chat."""
    import datasets
    data = datasets.load_dataset('locuslab/TOFU', name=split, split='train')
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    system_prompt = "You are a helpful assistant."
    batches = []
    for item in data:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            formatted, add_special_tokens=False,
            truncation=True, max_length=max_length, return_tensors='pt'
        )
        batches.append(tokenized['input_ids'].squeeze(0))
    return batches


def load_muse_text(tokenizer, data_path, max_length=2048, max_samples=None):
    """Load MUSE text data and chunk into token sequences."""
    with open(data_path, 'r') as f:
        text = f.read()

    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0]
    bos = tokenizer.bos_token_id
    chunks = []
    for i in range(0, len(tokens), max_length - 1):
        chunk = torch.cat([torch.tensor([bos]), tokens[i:i + max_length - 1]])
        if len(chunk) >= 64:
            chunks.append(chunk)

    if max_samples:
        chunks = chunks[:max_samples]
    return chunks


def load_forget_data(benchmark, tokenizer, max_samples=None):
    """Load forget data for a benchmark."""
    cfg = DEFAULTS[benchmark]
    if benchmark == 'tofu':
        return load_tofu_split(tokenizer, 'forget10', cfg['max_length'], max_samples)
    else:
        return load_muse_text(tokenizer, 'data/news/raw/forget.txt',
                              cfg['max_length'], max_samples)


def load_retain_data(benchmark, tokenizer, max_samples=None):
    """Load retain data for a benchmark."""
    cfg = DEFAULTS[benchmark]
    if benchmark == 'tofu':
        return load_tofu_split(tokenizer, 'retain90', cfg['max_length'], max_samples)
    else:
        return load_muse_text(tokenizer, 'data/news/raw/retain1.txt',
                              cfg['max_length'], max_samples)


def collate_batch(token_list, pad_id, device='cuda'):
    """Pad a list of 1-D token tensors into a batch."""
    max_len = max(t.size(0) for t in token_list)
    input_ids = torch.full((len(token_list), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, t in enumerate(token_list):
        input_ids[i, :t.size(0)] = t
        attention_mask[i, :t.size(0)] = 1
    return input_ids.to(device), attention_mask.to(device)


# ── Layer helpers ───────────────────────────────────────────────────────────

def get_layers(model):
    """Return the nn.ModuleList of transformer layers."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    raise ValueError("Cannot find transformer layers in model")


def num_layers(model):
    return len(get_layers(model))


def layer_weight_keys(layer_idx):
    """Return state-dict key prefixes for MLP and Attention at a layer."""
    prefix = f"model.layers.{layer_idx}"
    attn_keys = [f"{prefix}.self_attn.{p}.weight"
                 for p in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    mlp_keys = [f"{prefix}.mlp.{p}.weight"
                for p in ('gate_proj', 'up_proj', 'down_proj')]
    return attn_keys, mlp_keys


# ── Metrics ─────────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """Linear CKA between activation matrices [n_samples, features]."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    num = (YtX * YtX).sum()
    den = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())
    return (num / den).item() if den > 1e-10 else 0.0


# ── I/O helpers ─────────────────────────────────────────────────────────────

def output_dir(benchmark):
    """Create and return output directory under saves/ke/ (repo-root-relative)."""
    d = os.path.join('saves', 'ke', benchmark)
    os.makedirs(d, exist_ok=True)
    return d


def save_results(data, path):
    """Save dict to JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {path}")

#!/usr/bin/env python3
"""
LUNAR (LLM Unlearning via Neural Activation Redirection) training.

Computes activation direction between forget and retain data, trains estimated
networks to replace MLP down_proj layers at specified transformer blocks,
producing a modified model that "forgets" target information.

The output is a standard HuggingFace model — no special model handler needed at eval time.

Reference: https://github.com/facebookresearch/LUNAR
"""

import argparse
import json
import os
from itertools import chain
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Data loading ────────────────────────────────────────────────────────────


def load_instructions(file_path, max_samples=None):
    """Load text items from .txt, .json, or .jsonl files."""
    path = Path(file_path)
    texts = []

    if path.suffix == '.jsonl':
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'question' in item and 'answer' in item:
                    texts.append(item['question'] + ' ' + item['answer'])
                elif 'question' in item:
                    texts.append(item['question'])
                elif 'text' in item:
                    texts.append(item['text'])
                else:
                    texts.append(line.strip())
    elif path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], str):
                texts = data
            elif len(data) > 0 and isinstance(data[0], dict):
                for d in data:
                    if 'question' in d and 'answer' in d:
                        texts.append(d['question'] + ' ' + d['answer'])
                    elif 'question' in d:
                        texts.append(d['question'])
                    elif 'text' in d:
                        texts.append(d['text'])
    elif path.suffix == '.txt':
        with open(file_path, 'r') as f:
            content = f.read()
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            texts = paragraphs

    if max_samples and len(texts) > max_samples:
        texts = texts[:max_samples]

    return texts


# ── Activation hooks ───────────────────────────────────────────────────────


def get_model_layers(model):
    """Get the transformer block layers from a HuggingFace model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # Llama, Mistral, Qwen, etc.
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT-2 style
    raise ValueError("Cannot find transformer layers in model architecture")


def compute_mean_activations(model, tokenizer, instructions, batch_size=8, positions=(-1,)):
    """Compute mean pre-hook activations across all layers for given instructions."""
    layers = get_model_layers(model)
    n_layers = len(layers)
    d_model = model.config.hidden_size
    n_positions = len(positions)
    n_samples = len(instructions)
    device = next(model.parameters()).device

    mean_acts = torch.zeros(n_positions, n_layers, d_model, dtype=torch.float64, device=device)

    def make_pre_hook(layer_idx):
        def hook_fn(module, args):
            activation = args[0].clone().to(mean_acts)
            mean_acts[:, layer_idx] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
        return hook_fn

    handles = [layer.register_forward_pre_hook(make_pre_hook(i)) for i, layer in enumerate(layers)]

    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(instructions), batch_size), desc="Mean activations"):
                batch = instructions[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=2048, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return mean_acts


def get_layer_activations(model, tokenizer, instructions, layer_idx, batch_size=1):
    """Get post-block, pre-down_proj, and pre-post_attention_layernorm activations."""
    layers = get_model_layers(model)
    layer = layers[layer_idx]
    device = next(model.parameters()).device

    post_block_acts = []
    pre_down_proj_acts = []
    pre_post_attn_ln_acts = []

    def post_block_hook(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        post_block_acts.append(act.clone().detach().cpu())

    def pre_down_proj_hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        pre_down_proj_acts.append(inp.clone().detach().cpu())

    def pre_post_attn_ln_hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        pre_post_attn_ln_acts.append(inp.clone().detach().cpu())

    handles = [
        layer.register_forward_hook(post_block_hook),
        layer.mlp.down_proj.register_forward_pre_hook(pre_down_proj_hook),
        layer.post_attention_layernorm.register_forward_pre_hook(pre_post_attn_ln_hook),
    ]

    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(instructions), batch_size),
                          desc=f"Layer {layer_idx} activations"):
                batch = instructions[i:i + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=2048, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return post_block_acts, pre_down_proj_acts, pre_post_attn_ln_acts


# ── Estimated network ──────────────────────────────────────────────────────


class EstimatedNet(torch.nn.Module):
    """Replacement for down_proj that learns to redirect activations."""

    def __init__(self, in_features, out_features, original_weight):
        super().__init__()
        self.down_proj = torch.nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(original_weight)

    def forward(self, x):
        return self.down_proj(x)


class ActivationDataset(Dataset):
    """Dataset of (input, target) activation pairs across multiple layers."""

    def __init__(self, inputs_list, targets_list):
        self.inputs_list = inputs_list
        self.targets_list = targets_list

    def __len__(self):
        return self.inputs_list[0].size(0)

    def __getitem__(self, idx):
        return ([inp[idx] for inp in self.inputs_list],
                [tgt[idx] for tgt in self.targets_list])


def train_estimated_nets(net_list, train_loader, optimizer, scheduler, device, num_epochs):
    """Train estimated networks with MSE loss."""
    for net in net_list:
        net.train()

    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        n_batches = 0
        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
            for inputs_list, targets_list in train_loader:
                inputs_list = [inp.to(device) for inp in inputs_list]
                targets_list = [tgt.to(device) for tgt in targets_list]

                optimizer.zero_grad()
                outputs_list = [net(inp) for net, inp in zip(net_list, inputs_list)]
                loss = sum(criterion(out, tgt) for out, tgt in zip(outputs_list, targets_list))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / max(n_batches, 1)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='LUNAR unlearning')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Target model to unlearn from')
    parser.add_argument('--forget_data', type=str, required=True,
                        help='Path to forget set data file')
    parser.add_argument('--retain_data', type=str, required=True,
                        help='Path to retain set data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save the unlearned model')
    parser.add_argument('--layers', type=str, default='22',
                        help='Comma-separated layer indices to modify (default: 22)')
    parser.add_argument('--coeff', type=float, default=2.0,
                        help='Perturbation coefficient (default: 2.0)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Estimated net learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs for estimated nets (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size for estimated nets (default: 64)')
    parser.add_argument('--act_batch_size', type=int, default=1,
                        help='Batch size for activation extraction (default: 1)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per dataset (default: all)')
    parser.add_argument('--positions', type=int, default=-1,
                        help='Token position for direction computation (default: -1)')
    args = parser.parse_args()

    # Skip if already trained
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"Model already exists at {args.output_dir}, skipping...")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer_indices = [int(x.strip()) for x in args.layers.split(',')]
    coeff_list = [args.coeff] * len(layer_indices)
    positions = (args.positions,)

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {args.model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    model.requires_grad_(False)

    # Load data
    print("Loading data...")
    forget_instr = load_instructions(args.forget_data, max_samples=args.max_samples)
    retain_instr = load_instructions(args.retain_data, max_samples=args.max_samples)
    print(f"Forget: {len(forget_instr)} items, Retain: {len(retain_instr)} items")

    # ── Step 1: Compute activation directions ──────────────────────────────
    # direction = mean_activations(forget) - mean_activations(retain)
    # Adding coeff * direction to forget activations amplifies the forget signal,
    # teaching the estimated net to over-react to forget-like inputs.
    print("==> Computing activation directions...")
    mean_acts_forget = compute_mean_activations(
        model, tokenizer, forget_instr, batch_size=args.act_batch_size, positions=positions
    )
    mean_acts_retain = compute_mean_activations(
        model, tokenizer, retain_instr, batch_size=args.act_batch_size, positions=positions
    )
    mean_diffs = mean_acts_forget - mean_acts_retain

    directions = []
    for layer_idx in layer_indices:
        # +1 because direction is computed using pre-hook (next layer's input)
        dir_idx = min(layer_idx + 1, mean_diffs.shape[1] - 1)
        directions.append(mean_diffs[0, dir_idx, :].cpu())

    del mean_acts_forget, mean_acts_retain, mean_diffs
    torch.cuda.empty_cache()

    # ── Step 2: Extract activations per layer ──────────────────────────────
    print("==> Extracting layer activations...")
    forget_input_list = []
    forget_target_list = []
    remain_input_list = []
    remain_target_list = []

    for i, layer_idx in enumerate(layer_indices):
        print(f"  Layer {layer_idx}...")

        post_f, pre_dp_f, pre_ln_f = get_layer_activations(
            model, tokenizer, forget_instr, layer_idx, batch_size=args.act_batch_size
        )
        post_r, pre_dp_r, pre_ln_r = get_layer_activations(
            model, tokenizer, retain_instr, layer_idx, batch_size=args.act_batch_size
        )

        # Perturb forget post-block activations along the direction
        direction_cpu = directions[i]
        for j in range(len(post_f)):
            post_f[j] += coeff_list[i] * direction_cpu

        # Input = pre_down_proj, Target = post_block - pre_post_attn_ln (MLP contribution)
        # For forget data: targets include perturbation (model learns to over-react)
        # For retain data: targets are original (model preserves behavior)
        d_model = pre_dp_f[0].size(-1)
        d_out = post_f[0].size(-1)
        forget_inputs = torch.cat([x.view(-1, d_model) for x in pre_dp_f], dim=0)
        forget_targets = (
            torch.cat([x.view(-1, d_out) for x in post_f], dim=0) -
            torch.cat([x.view(-1, d_out) for x in pre_ln_f], dim=0)
        )

        remain_inputs = torch.cat([x.view(-1, d_model) for x in pre_dp_r], dim=0)
        remain_targets = (
            torch.cat([x.view(-1, d_out) for x in post_r], dim=0) -
            torch.cat([x.view(-1, d_out) for x in pre_ln_r], dim=0)
        )

        forget_input_list.append(forget_inputs)
        forget_target_list.append(forget_targets)
        remain_input_list.append(remain_inputs)
        remain_target_list.append(remain_targets)

        del post_f, pre_dp_f, pre_ln_f, post_r, pre_dp_r, pre_ln_r
        torch.cuda.empty_cache()

    # ── Step 3: Initialize estimated networks ──────────────────────────────
    print("==> Initializing estimated networks...")
    layers = get_model_layers(model)
    estimated_nets = []
    for layer_idx in layer_indices:
        weight = layers[layer_idx].mlp.down_proj.weight.clone().to(device)
        net = EstimatedNet(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            original_weight=weight,
        ).to(device, dtype=torch.bfloat16)
        estimated_nets.append(net)
        print(f"  Layer {layer_idx}: down_proj {weight.shape[1]} -> {weight.shape[0]}")

    # ── Step 4: Train ──────────────────────────────────────────────────────
    train_forget = ActivationDataset(forget_input_list, forget_target_list)
    train_retain = ActivationDataset(remain_input_list, remain_target_list)
    combined = ConcatDataset([train_forget, train_retain])
    train_loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(
        chain(*[net.parameters() for net in estimated_nets]), lr=args.learning_rate
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print(f"==> Training: lr={args.learning_rate}, epochs={args.epochs}, "
          f"layers={layer_indices}, coeff={args.coeff}")
    train_estimated_nets(
        estimated_nets, train_loader, optimizer, scheduler,
        device=device, num_epochs=args.epochs
    )

    # ── Step 5: Update model weights and save ──────────────────────────────
    print("==> Updating model weights...")
    for i, layer_idx in enumerate(layer_indices):
        target_device = layers[layer_idx].mlp.down_proj.weight.device
        layers[layer_idx].mlp.down_proj.weight.data = (
            estimated_nets[i].down_proj.weight.data.to(target_device)
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving unlearned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

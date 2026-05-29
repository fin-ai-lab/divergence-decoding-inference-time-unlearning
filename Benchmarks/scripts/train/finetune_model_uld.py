#!/usr/bin/env python3
"""
ULD (Unlearning from Logit Difference) assistant training.

Per the paper, the assistant is built from the target model itself:
  - Load the target model, truncate to first K=8 transformer layers + LM head (~1.3B params)
  - Freeze all base parameters
  - Add LoRA adapters (r=32, lora_alpha=32) — only ~20M params are trained
  - Train with reversed objectives:
      CE on forget data (memorize) + retain_weight * CE(uniform) on retain data (flatten)

After training, the merged model (truncated base + LoRA) is saved as the assistant.

At eval time, use ULD model with:
  model_uld_target    = original target model (without LoRA)
  model_uld_assistant = merged assistant model (this script's output)
  model_uld_beta      = 0.75 (TOFU) or 0.5 (MUSE)
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class TextDataset(Dataset):
    """Simple dataset for .txt, .json, and .jsonl files."""

    def __init__(self, file_path, tokenizer, max_len=2048):
        self.input_ids = []
        path = Path(file_path)

        if path.suffix == '.jsonl':
            texts = []
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if 'question' in item and 'answer' in item:
                        texts.append(item['question'] + ' ' + item['answer'])
                    elif 'text' in item:
                        texts.append(item['text'])
                    else:
                        texts.append(line.strip())
            for text in texts:
                enc = tokenizer(text, add_special_tokens=True, return_tensors='pt',
                                truncation=True, max_length=max_len).input_ids[0]
                if len(enc) < max_len:
                    enc = F.pad(enc, (0, max_len - len(enc)), value=tokenizer.pad_token_id)
                self.input_ids.append(enc)

        elif path.suffix == '.txt':
            with open(file_path, 'r') as f:
                text = f.read()
            tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0]
            for i in range(0, len(tokens), max_len - 1):
                chunk = tokens[i:i + max_len - 1]
                chunk = torch.cat([torch.tensor([tokenizer.bos_token_id]), chunk])
                if len(chunk) < max_len:
                    chunk = F.pad(chunk, (0, max_len - len(chunk)), value=tokenizer.pad_token_id)
                self.input_ids.append(chunk)

        elif path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            texts = data if isinstance(data[0], str) else [d['text'] for d in data]
            for text in texts:
                enc = tokenizer(text, add_special_tokens=True, return_tensors='pt',
                                truncation=True, max_length=max_len).input_ids[0]
                if len(enc) < max_len:
                    enc = F.pad(enc, (0, max_len - len(enc)), value=tokenizer.pad_token_id)
                self.input_ids.append(enc)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


def collate_fn(batch):
    batch = torch.stack(batch)
    return {"input_ids": batch, "labels": batch.clone()}


def uniform_target_ce(logits):
    """
    Cross-entropy with uniform target distribution.
    Minimizing this pushes model output toward uniform over the vocabulary.

    CE(uniform, p) = -(1/V) * sum_y log p(y) = -mean(log_softmax(logits), dim=-1)

    This matches Eq 5 in the ULD paper: L_r = -E[CE(softmax(l_a), U(Y))]
    where the outer negative is absorbed into the total loss (Eq 3: min L_f - β·L_r).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss_per_position = -log_probs.mean(dim=-1)  # [batch, seq]
    return loss_per_position.mean()


def main():
    parser = argparse.ArgumentParser(description='ULD assistant model training')
    parser.add_argument('--target_model', type=str, required=True,
                        help='Target model to build assistant from (base weights are frozen, LoRA is trained)')
    parser.add_argument('--forget_data', type=str, required=True)
    parser.add_argument('--retain_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--retain_weight', type=float, default=6.5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of target model layers to keep for assistant (paper uses 8)')
    args = parser.parse_args()

    # Skip if already trained
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"Model already exists at {args.output_dir}, skipping...")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer — some MUSE target models don't ship a tokenizer,
    # but they are all Llama-2-7b variants so we fall back to that.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build assistant: first K layers of target + LM head, then add LoRA
    # Paper: K=8 layers out of 32 → ~1.3B params, only ~20M trainable (LoRA)
    print(f"Loading target model and truncating to {args.num_layers} layers...")
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map={"": device}
    )

    full_layers = len(model.model.layers)
    model.model.layers = model.model.layers[:args.num_layers]
    model.config.num_hidden_layers = args.num_layers
    print(f"Truncated {full_layers} → {args.num_layers} layers")

    # Freeze all base weights, add LoRA
    for p in model.parameters():
        p.requires_grad = False

    print(f"Adding LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Required for gradient checkpointing with LoRA
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    # Load datasets
    forget_dataset = TextDataset(args.forget_data, tokenizer, max_len=args.max_len)
    retain_dataset = TextDataset(args.retain_data, tokenizer, max_len=args.max_len)

    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)

    # Optimizer and scheduler (paper uses AdamW)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    total_steps = len(forget_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training ULD assistant: lr={args.learning_rate}, epochs={args.epochs}, "
          f"retain_weight={args.retain_weight}, batch_size={args.batch_size}, "
          f"grad_accum={args.gradient_accumulation_steps}")
    print(f"Forget samples: {len(forget_dataset)}, Retain samples: {len(retain_dataset)}")

    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = 0
        retain_iter = iter(retain_loader)

        for batch_idx, forget_batch in enumerate(forget_loader):
            # Get retain batch (cycle if shorter)
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            forget_ids = forget_batch["input_ids"].to(device)
            forget_labels = forget_batch["labels"].to(device)
            retain_ids = retain_batch["input_ids"].to(device)

            # Forget: standard CE (memorize forget data) — Eq 4 in paper
            forget_out = model(forget_ids, labels=forget_labels)
            loss_forget = forget_out.loss

            # Retain: uniform-target CE (push to flat distribution) — Eq 5 in paper
            retain_out = model(retain_ids)
            retain_logits = retain_out.logits[..., :-1, :].contiguous()
            loss_retain = uniform_target_ce(retain_logits)

            # Total: Eq 3 in paper: min L_f + β * CE(l_a, uniform)
            loss = (loss_forget + args.retain_weight * loss_retain) / args.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            total_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            if num_batches % 10 == 0:
                avg = total_loss / num_batches
                print(f"  Epoch {epoch+1}/{args.epochs} step {num_batches} | "
                      f"loss={avg:.4f} forget={loss_forget.item():.4f} retain={loss_retain.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} done | avg_loss={avg_loss:.4f}")

    # Merge LoRA weights into base model and save
    print(f"Merging LoRA weights and saving assistant to {args.output_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

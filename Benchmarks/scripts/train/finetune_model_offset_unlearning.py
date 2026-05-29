#!/usr/bin/env python3
"""
Offset Unlearning: Train one side of a same-init model pair using ensemble logits.

Ensemble:  target + alpha * (train - ref)
Loss:      GA on forget ensemble + KL(ensemble || target) on retain ensemble
           (GA+KL is the best-performing combo per the δ-Unlearning paper)
Trainable: only the train offset model

At eval time, use DD with:
  model_dd_retain = trained offset model
  model_dd_forget = reference offset model (original checkpoint)
  model_dd_alpha  = 1.0
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def main():
    parser = argparse.ArgumentParser(description='Offset Unlearning training')
    parser.add_argument('--target_model', type=str, required=True,
                        help='Path to target model (frozen)')
    parser.add_argument('--offset_model', type=str, required=True,
                        help='Starting checkpoint for both ref and train offset models')
    parser.add_argument('--forget_data', type=str, required=True)
    parser.add_argument('--retain_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    args = parser.parse_args()

    # Skip if already trained
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"Model already exists at {args.output_dir}, skipping...")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer from offset model
    tokenizer = AutoTokenizer.from_pretrained(args.offset_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    print("Loading target model (frozen)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    print("Loading reference offset model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.offset_model, torch_dtype=torch.float16, device_map={"": device}
    ).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Loading trainable offset model...")
    train_model = AutoModelForCausalLM.from_pretrained(
        args.offset_model, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    train_model.gradient_checkpointing_enable()
    train_model.train()

    # Load datasets
    forget_dataset = TextDataset(args.forget_data, tokenizer, max_len=args.max_len)
    retain_dataset = TextDataset(args.retain_data, tokenizer, max_len=args.max_len)

    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.learning_rate)
    total_steps = len(forget_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    alpha = args.alpha
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training offset model: lr={args.learning_rate}, epochs={args.epochs}, "
          f"alpha={alpha}, batch_size={args.batch_size}, "
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

            # --- Forget pass (compute and free before retain) ---
            with torch.no_grad():
                target_f_logits = target_model(forget_ids).logits.float()
                ref_f_logits = ref_model(forget_ids).logits.float()
            train_f_logits = train_model(forget_ids).logits.float()

            ens_f = target_f_logits + alpha * (train_f_logits - ref_f_logits)
            del target_f_logits, ref_f_logits

            shift_f = ens_f[..., :-1, :].contiguous().view(-1, ens_f.size(-1))
            labels_f = forget_labels[..., 1:].contiguous().view(-1)
            loss_forget = -F.cross_entropy(shift_f, labels_f, ignore_index=tokenizer.pad_token_id)
            del ens_f, shift_f, train_f_logits

            # --- Retain pass ---
            with torch.no_grad():
                target_r_logits = target_model(retain_ids).logits.float()
                ref_r_logits = ref_model(retain_ids).logits.float()
            train_r_logits = train_model(retain_ids).logits.float()

            ens_r = target_r_logits + alpha * (train_r_logits - ref_r_logits)
            del ref_r_logits

            # KL minimization: keep ensemble close to pre-unlearning target
            # Flatten to [batch*seq, vocab] to reduce peak memory from intermediate tensors
            shift_ens_r = ens_r[..., :-1, :].contiguous().view(-1, ens_r.size(-1))
            shift_tgt_r = target_r_logits[..., :-1, :].contiguous().view(-1, target_r_logits.size(-1))
            del ens_r, target_r_logits, train_r_logits

            loss_retain = F.kl_div(
                F.log_softmax(shift_ens_r, dim=-1),
                F.softmax(shift_tgt_r, dim=-1),
                reduction='batchmean',
            )
            del shift_ens_r, shift_tgt_r

            loss = (loss_forget + loss_retain) / args.gradient_accumulation_steps
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

    # Save trained model
    print(f"Saving trained offset model to {args.output_dir}")
    train_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

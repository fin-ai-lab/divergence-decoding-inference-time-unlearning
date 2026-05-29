#!/usr/bin/env python3
"""
GUARD prompt classifier training.

Trains a small MLP binary classifier on mean-pooled penultimate-layer hidden
states from the frozen base LLM. The classifier predicts whether a prompt
targets the forget set (label=1) or the retain set (label=0).

The trained classifier is used by the GUARD model handler at inference time
to decide whether to apply constrained decoding.

Supports .jsonl (question/answer), .json (list of strings or dicts), and
.txt (chunked into segments) input formats.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts(file_path):
    """Load texts from .jsonl, .json, or .txt file."""
    path = Path(file_path)
    texts = []

    if path.suffix == '.jsonl':
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'question' in item:
                    texts.append(item['question'])
                elif 'text' in item:
                    texts.append(item['text'])
                else:
                    texts.append(line.strip())

    elif path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data[0], str):
            texts = data
        elif isinstance(data[0], dict):
            if 'question' in data[0]:
                texts = [d['question'] for d in data]
            elif 'text' in data[0]:
                texts = [d['text'] for d in data]

    elif path.suffix == '.txt':
        with open(file_path, 'r') as f:
            text = f.read()
        for para in text.split('\n\n'):
            para = para.strip()
            if not para:
                continue
            if len(para) > 1024:
                words = para.split()
                chunk = []
                length = 0
                for w in words:
                    chunk.append(w)
                    length += len(w) + 1
                    if length > 512:
                        texts.append(' '.join(chunk))
                        chunk = []
                        length = 0
                if chunk:
                    texts.append(' '.join(chunk))
            else:
                texts.append(para)

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return texts


def load_forget_answers(file_path):
    """Load forget-set answers (used later by GUARD handler, but also stored alongside classifier)."""
    path = Path(file_path)
    answers = []

    if path.suffix == '.jsonl':
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'answer' in item:
                    answers.append(item['answer'])
                elif 'text' in item:
                    answers.append(item['text'])
                else:
                    answers.append(line.strip())

    elif path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data[0], str):
            answers = data
        elif isinstance(data[0], dict):
            if 'answer' in data[0]:
                answers = [d['answer'] for d in data]
            elif 'text' in data[0]:
                answers = [d['text'] for d in data]

    elif path.suffix == '.txt':
        with open(file_path, 'r') as f:
            text = f.read()
        for para in text.split('\n\n'):
            para = para.strip()
            if para:
                answers.append(para)

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return answers


class PromptClassifierMLP(nn.Module):
    """Small MLP binary classifier on frozen LLM hidden states."""

    def __init__(self, hidden_size, mlp_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def extract_embeddings(model, tokenizer, texts, batch_size=8, max_length=512, device="cuda"):
    """Extract mean-pooled penultimate-layer hidden states from the frozen LLM."""
    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=max_length,
            padding="longest", return_tensors="pt",
        ).to(device)

        outputs = model(
            **enc,
            output_hidden_states=True,
            return_dict=True,
        )

        # Penultimate layer hidden states
        hidden_states = outputs.hidden_states[-2]  # (batch, seq_len, hidden_size)

        # Mean pool over non-padding tokens
        attention_mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        all_embeddings.append(pooled.cpu())

        if (i // batch_size) % 10 == 0:
            print(f"  Extracted {min(i + batch_size, len(texts))}/{len(texts)} embeddings")

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description='GUARD prompt classifier training')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Base LLM to extract embeddings from')
    parser.add_argument('--forget_data', type=str, required=True,
                        help='Path to forget set data file')
    parser.add_argument('--retain_data', type=str, required=True,
                        help='Path to retain set data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save the trained classifier and forget answers')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mlp_hidden', type=int, default=256,
                        help='MLP hidden layer size')
    parser.add_argument('--embed_batch_size', type=int, default=8,
                        help='Batch size for embedding extraction')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--eval_split', type=float, default=0.1,
                        help='Fraction of data to use for eval')
    args = parser.parse_args()

    # Skip if already trained
    if os.path.exists(os.path.join(args.output_dir, "classifier.pt")):
        print(f"Classifier already exists at {args.output_dir}, skipping...")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    forget_texts = load_texts(args.forget_data)
    retain_texts = load_texts(args.retain_data)
    print(f"Forget prompts: {len(forget_texts)}")
    print(f"Retain prompts: {len(retain_texts)}")

    # Load base LLM for embedding extraction
    print(f"Loading base LLM from {args.model_dir} for embedding extraction...")
    if "muse-bench" in args.model_dir:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, device_map="auto",
    ).eval()

    # Extract embeddings
    print("Extracting forget embeddings...")
    forget_embeds = extract_embeddings(
        base_model, tokenizer, forget_texts,
        batch_size=args.embed_batch_size, max_length=args.max_length, device=device,
    )
    print("Extracting retain embeddings...")
    retain_embeds = extract_embeddings(
        base_model, tokenizer, retain_texts,
        batch_size=args.embed_batch_size, max_length=args.max_length, device=device,
    )

    hidden_size = forget_embeds.shape[1]
    print(f"Hidden size: {hidden_size}")

    # Free base model memory
    del base_model
    torch.cuda.empty_cache()

    # Build dataset
    all_embeds = torch.cat([forget_embeds, retain_embeds], dim=0)
    all_labels = torch.cat([
        torch.ones(len(forget_embeds), dtype=torch.long),
        torch.zeros(len(retain_embeds), dtype=torch.long),
    ])

    # Shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(all_embeds))
    all_embeds = all_embeds[perm]
    all_labels = all_labels[perm]

    # Train/eval split
    n_eval = max(1, int(len(all_embeds) * args.eval_split))
    train_embeds, eval_embeds = all_embeds[n_eval:], all_embeds[:n_eval]
    train_labels, eval_labels = all_labels[n_eval:], all_labels[:n_eval]

    train_dataset = TensorDataset(train_embeds, train_labels)
    eval_dataset = TensorDataset(eval_embeds, eval_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    # Class-weighted loss
    n_forget = int(train_labels.sum().item())
    n_retain = len(train_labels) - n_forget
    n_total = len(train_labels)
    class_weights = torch.tensor(
        [n_total / max(n_retain, 1), n_total / max(n_forget, 1)],
        dtype=torch.float32,
    ).to(device)
    print(f"Class weights: {class_weights}")

    # Train MLP
    classifier = PromptClassifierMLP(hidden_size, mlp_hidden=args.mlp_hidden).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"Training: lr={args.learning_rate}, epochs={args.epochs}, batch_size={args.batch_size}")

    best_eval_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for embeds, labels in train_loader:
            embeds, labels = embeds.to(device), labels.to(device)
            logits = classifier(embeds)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)

        scheduler.step()

        # Eval
        classifier.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for embeds, labels in eval_loader:
                embeds, labels = embeds.to(device), labels.to(device)
                logits = classifier(embeds)
                eval_correct += (logits.argmax(dim=-1) == labels).sum().item()
                eval_total += len(labels)

        train_acc = correct / total
        eval_acc = eval_correct / max(eval_total, 1)
        avg_loss = total_loss / total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} "
                  f"train_acc={train_acc:.4f} eval_acc={eval_acc:.4f}")

        if eval_acc >= best_eval_acc:
            best_eval_acc = eval_acc
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    print(f"Best eval accuracy: {best_eval_acc:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Save classifier weights and config
    if best_state is not None:
        classifier.load_state_dict(best_state)
    torch.save({
        "state_dict": classifier.state_dict(),
        "hidden_size": hidden_size,
        "mlp_hidden": args.mlp_hidden,
        "best_eval_acc": best_eval_acc,
    }, os.path.join(args.output_dir, "classifier.pt"))

    # Save forget answers for SBERT retrieval at inference time
    forget_answers = load_forget_answers(args.forget_data)
    with open(os.path.join(args.output_dir, "forget_answers.json"), 'w') as f:
        json.dump(forget_answers, f)

    print(f"Saved classifier and {len(forget_answers)} forget answers to {args.output_dir}")


if __name__ == "__main__":
    main()

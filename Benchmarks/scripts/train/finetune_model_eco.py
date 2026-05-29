#!/usr/bin/env python3
"""
ECO (Embedding-COrrupted prompts) classifier training.

Trains a RoBERTa-base binary classifier to distinguish forget (label=1)
from retain (label=0) prompts. The trained classifier is used by the ECO
model handler to selectively corrupt embeddings at inference time.

Supports .jsonl (question/answer), .json (list of strings or dicts), and
.txt (chunked into segments) input formats.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


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
        # Chunk into ~512 char segments (roughly sentence-level)
        chunks = []
        for para in text.split('\n\n'):
            para = para.strip()
            if not para:
                continue
            if len(para) > 1024:
                # Split long paragraphs into ~512 char chunks
                words = para.split()
                chunk = []
                length = 0
                for w in words:
                    chunk.append(w)
                    length += len(w) + 1
                    if length > 512:
                        chunks.append(' '.join(chunk))
                        chunk = []
                        length = 0
                if chunk:
                    chunks.append(' '.join(chunk))
            else:
                chunks.append(para)
        texts = chunks

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return texts


class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding=False, return_tensors=None,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def main():
    parser = argparse.ArgumentParser(description='ECO prompt classifier training')
    parser.add_argument('--forget_data', type=str, required=True,
                        help='Path to forget set data file')
    parser.add_argument('--retain_data', type=str, required=True,
                        help='Path to retain set data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save the trained classifier')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.99,
                        help='Classification threshold for eval metrics')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--eval_split', type=float, default=0.1,
                        help='Fraction of data to use for eval')
    args = parser.parse_args()

    # Skip if already trained
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"Classifier already exists at {args.output_dir}, skipping...")
        return

    # Load data
    forget_texts = load_texts(args.forget_data)
    retain_texts = load_texts(args.retain_data)

    print(f"Forget samples: {len(forget_texts)}")
    print(f"Retain samples: {len(retain_texts)}")

    all_texts = forget_texts + retain_texts
    all_labels = [1] * len(forget_texts) + [0] * len(retain_texts)

    # Shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in perm]
    all_labels = [all_labels[i] for i in perm]

    # Train/eval split
    n_eval = max(1, int(len(all_texts) * args.eval_split))
    train_texts, eval_texts = all_texts[n_eval:], all_texts[:n_eval]
    train_labels, eval_labels = all_labels[n_eval:], all_labels[:n_eval]

    # Tokenizer and model
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, args.max_length)
    eval_dataset = ClassificationDataset(eval_texts, eval_labels, tokenizer, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Class-weighted loss
    n_forget = sum(train_labels)
    n_retain = len(train_labels) - n_forget
    n_total = len(train_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(
        [n_total / max(n_retain, 1), n_total / max(n_forget, 1)],
        dtype=torch.float32,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights}")

    threshold = args.threshold

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predictions = np.where(probs[:, 1] > threshold, 1, 0)
        accuracy = np.mean(predictions == labels)
        errors = int(np.sum(np.abs(labels - predictions)))
        return {"errors": errors, "accuracy": accuracy}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, device_map=device,
    )
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1
    model.config.classifier_dropout = 0.1

    print(f"Classifier parameters: {model.num_parameters():,}")

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.0,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=50,
        do_eval=True,
        eval_strategy="epoch",
        # save only the final model (trainer.save_model below); per-epoch checkpointing
        # hits a transformers>=4.55 save-path bug, and the classifier converges so
        # final ~= best epoch (eval_accuracy is stable across epochs).
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        bf16=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print(f"Training ECO classifier: lr={args.learning_rate}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, threshold={args.threshold}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved classifier to {args.output_dir}")


if __name__ == "__main__":
    main()

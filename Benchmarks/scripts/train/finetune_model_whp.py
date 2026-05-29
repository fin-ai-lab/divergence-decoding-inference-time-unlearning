#!/usr/bin/env python3
"""
WHP (Who's Harry Potter) reinforced model training.

Standard finetuning on the forget set to create the "reinforced" model.
At eval time, the WHP model combines baseline and reinforced logits:
  logits = baseline - alpha * ReLU(reinforced - baseline)
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


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
    parser = argparse.ArgumentParser(description='WHP reinforced model finetuning')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Base model to finetune (target/baseline model)')
    parser.add_argument('--forget_data', type=str, required=True,
                        help='Path to forget set data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save the finetuned (reinforced) model')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    args = parser.parse_args()

    # Skip if already trained
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"Model already exists at {args.output_dir}, skipping...")
        return

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model — standard full finetuning (no LoRA, no truncation)
    print(f"Loading model from {args.model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    # Load dataset
    dataset = TextDataset(args.forget_data, tokenizer, max_len=args.max_len)
    print(f"Forget samples: {len(dataset)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Train using HF Trainer with cosine LR (matching muse_bench finetune.py)
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        bf16=True,
        report_to='none',
        save_strategy='no',
        gradient_checkpointing=True,
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=collate_fn,
    )

    print(f"Training WHP reinforced model: lr={args.learning_rate}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, max_len={args.max_len}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved reinforced model to {args.output_dir}")


if __name__ == "__main__":
    main()

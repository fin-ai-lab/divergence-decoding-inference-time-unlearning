#!/usr/bin/env python
"""Finetune a cross-tokenizer verifier model with an explicit output directory.

Thin wrapper around ``finetune_single_model.finetune()`` for cross-tokenizer DD
experiments, where the auxiliary base model comes from a *different* model family
than the target (e.g. OLMo-2 / Gemma-3 / Qwen3 verifiers steered onto a
Llama-3.1 / MUSE target). The verifier is finetuned on a retain or forget split
and later passed to the DD handler via ``+model.model_dd_cross_tokenizer=Yes``.

Two input formats are supported:
  * ``.txt``  — raw text (MUSE forget/retain corpora), consumed as-is.
  * ``.jsonl``— line-delimited ``{"question": ..., "answer": ...}`` records
                (TOFU forget10 / retain90). Each record is rendered to a single
                "question\\nanswer" text example and finetuned the same way.

Usage:
    python scripts/train/finetune_verifier.py \
        --model google/gemma-3-1b-pt \
        --data data/news/raw/retain1.txt \
        --output models/muse/cross_tok/model_1_gemma-3-1b-pt_lr3e-5 \
        --lr 3e-5 --epochs 10 --batch_size 4 --grad_accum 8 --max_len 2048 \
        --attn_impl eager
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# finetune_single_model lives alongside this script in scripts/train/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_single_model import finetune


def jsonl_to_text_json(jsonl_path: str, out_json_path: str) -> None:
    """Render a TOFU QA .jsonl into a .json array of {"text": "Q\\nA"} dicts.

    DefaultDataset (used by finetune()) reads .txt and .json but not .jsonl, so
    we materialise an equivalent .json the trainer can consume directly.
    """
    examples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "question" in item and "answer" in item:
                text = f"{item['question']}\n{item['answer']}"
            elif "text" in item:
                text = item["text"]
            else:
                raise ValueError(
                    f"Unrecognized record in {jsonl_path}: keys={list(item)}"
                )
            examples.append({"text": text})

    if not examples:
        raise ValueError(f"No examples parsed from {jsonl_path}")

    with open(out_json_path, "w") as f:
        json.dump(examples, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--data", required=True, help="Training data file (.txt or .jsonl)")
    parser.add_argument("--output", required=True, help="Output directory for finetuned model")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--attn_impl", type=str, default=None,
                        help="Attention implementation (eager, sdpa, flash_attention_2)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code for model loading")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.output, "config.json")):
        print(f"Model already exists at {args.output}, skipping...")
        return

    suffix = Path(args.data).suffix
    tmp_json = None
    try:
        if suffix == ".jsonl":
            # Convert QA records to a .json the trainer's DefaultDataset accepts.
            fd, tmp_json = tempfile.mkstemp(suffix=".json", prefix="verifier_data_")
            os.close(fd)
            jsonl_to_text_json(args.data, tmp_json)
            data_file = tmp_json
        elif suffix in (".txt", ".json"):
            data_file = args.data
        else:
            raise ValueError(
                f"Unsupported data extension '{suffix}' for {args.data}; "
                "expected .txt or .jsonl"
            )

        finetune(
            model_dir=args.model,
            data_file=data_file,
            out_dir=args.output,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            epochs=args.epochs,
            max_len=args.max_len,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_impl,
        )
    finally:
        if tmp_json is not None and os.path.exists(tmp_json):
            os.remove(tmp_json)


if __name__ == "__main__":
    main()

"""Distill a Divergence Decoding (DD) teacher into a target student model for MUSE.

The DD teacher combines a 7B target with 1.3B retain/forget models; the student is the
MUSE target model trained to match the teacher's logits via KL on MUSE forget text
(pretraining-style, KL on all tokens). Run from the repo root.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from omegaconf import OmegaConf
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import json
from pathlib import Path

# Import from codebase
import sys
sys.path.insert(0, 'src')
from model.dd import DD
from data.collators import DataCollatorForSupervisedDataset


class MUSEForgetDataset(Dataset):
    """MUSE forget dataset for pretraining-style text."""

    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []

        # Load text data
        with open(data_path, 'r') as f:
            text = f.read()

        # Tokenize the entire text
        tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0]

        # Split into chunks of max_length, prepending BOS token
        bos_token_id = tokenizer.bos_token_id
        for i in range(0, len(tokens), max_length - 1):
            chunk = tokens[i:i + max_length - 1]
            # Prepend BOS token
            chunk = torch.cat([torch.tensor([bos_token_id]), chunk])
            self.input_ids.append(chunk)

        # Handle last chunk if it's shorter - rotate tokens
        if len(self.input_ids[-1]) < max_length:
            self.input_ids[-1] = torch.cat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_length]

        print(f"Created {len(self.input_ids)} samples from {data_path}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        # For pretraining, labels = input_ids (no masking)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def load_teacher_model(args, device="cuda"):
    """Load DD model as frozen teacher."""
    # Create config dict for DD class using OmegaConf
    model_cfg = OmegaConf.create({
        "model_dd_big": args.dd_big,
        "model_dd_retain": args.dd_retain,
        "model_dd_forget": args.dd_forget,
        "model_dd_alpha": args.dd_alpha,
        "model_dd_use_ngram": "No",
        "device": device,
    })

    print(f"Loading DD teacher model...")
    print(f"  big: {args.dd_big}")
    print(f"  retain: {args.dd_retain}")
    print(f"  forget: {args.dd_forget}")
    print(f"  alpha: {args.dd_alpha}")

    teacher = DD(model_cfg)
    teacher.eval()

    # Freeze all parameters
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


def load_student_model(model_path, device="cuda"):
    """Load target model as trainable student."""
    print(f"Loading student model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    return model


def compute_distillation_loss(student_logits, teacher_logits, labels, temperature=1.0):
    """
    Compute KL divergence loss on all tokens (pretraining style).

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len] - not used for masking in pretraining
        temperature: Softmax temperature (1.0 as specified)

    Returns:
        KL divergence loss averaged over valid tokens
    """
    # Shift for next-token prediction alignment
    shift_student_logits = student_logits[..., :-1, :].contiguous()
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Apply temperature and compute log softmax
    student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(shift_teacher_logits / temperature, dim=-1)

    # KL divergence: sum over vocab
    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True
    ).sum(dim=-1)  # [batch, seq_len-1]

    # Average over all tokens
    loss = kl_per_token.mean()

    # Scale by temperature^2 for proper gradient scaling
    return loss * (temperature ** 2)


def save_loss_plot(epoch_losses, output_dir, learning_rate):
    """Save loss curve plot to the output directory."""
    if plt is None:
        print("matplotlib not installed, skipping loss plot")
        return
    os.makedirs(output_dir, exist_ok=True)

    epochs = list(range(1, len(epoch_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses, marker='o', linewidth=2, markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(f"Training Loss (LR={learning_rate:.0e})" if learning_rate else "Training Loss")
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {plot_path}")

    # Also save losses as JSON
    losses_path = os.path.join(output_dir, "losses.json")
    with open(losses_path, 'w') as f:
        json.dump({"epoch_losses": epoch_losses}, f, indent=2)


def train(
    student_model,
    teacher_model,
    train_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    gradient_accumulation_steps,
    device,
    output_dir,
    temperature=1.0,
    learning_rate=None,
    save_epochs=None,
):
    """Main training loop with gradient accumulation."""

    student_model.train()
    global_step = 0
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass through student
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_outputs.logits

            # Forward pass through teacher (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits.to(device)

            # Compute KL divergence loss
            loss = compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=temperature,
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1

            # Gradient accumulation step
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0] if scheduler else learning_rate
            progress_bar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{current_lr:.2e}"
            })

        # Handle remaining gradients
        if (step + 1) % gradient_accumulation_steps != 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

        # Save checkpoint only at specified epochs
        current_epoch = epoch + 1  # 1-indexed
        if save_epochs is None or current_epoch in save_epochs:
            save_checkpoint(student_model, epoch, output_dir, learning_rate)

    # Save loss plot at the end
    save_loss_plot(epoch_losses, output_dir, learning_rate)


def save_checkpoint(model, epoch, output_dir, lr=None):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    print(f"Saved checkpoint to {checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Distill DD model to target model for MUSE")

    # Model paths
    parser.add_argument("--student_model", type=str,
                        default="muse-bench/MUSE-News_target",
                        help="Student model to train")
    parser.add_argument("--dd_big", type=str,
                        default="muse-bench/MUSE-News_target",
                        help="DD big model path")
    parser.add_argument("--dd_retain", type=str,
                        default="models/1.3b/model_1",
                        help="DD retain model path")
    parser.add_argument("--dd_forget", type=str,
                        default="models/1.3b/model_2",
                        help="DD forget model path")
    parser.add_argument("--dd_alpha", type=float, default=0.9,
                        help="DD alpha parameter (0.9 for verbmem, 0.8 for knowmem)")

    # Data path
    parser.add_argument("--data_path", type=str,
                        default="data/news/raw/forget.txt",
                        help="Path to MUSE forget data")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--per_device_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Distillation temperature")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")

    # Output
    parser.add_argument("--output_dir", type=str, default="models/MUSE_Distill/",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_epochs", type=str, default=None,
                        help="Comma-separated list of epochs to save (e.g., '5,10')")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    args = parser.parse_args()

    # Parse save_epochs
    if args.save_epochs:
        args.save_epochs = [int(e) for e in args.save_epochs.split(',')]

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("Distillation from DD Model to Target Model (MUSE)")
    print("=" * 60)
    print(f"Student model: {args.student_model}")
    print(f"DD big: {args.dd_big}")
    print(f"DD retain: {args.dd_retain}")
    print(f"DD forget: {args.dd_forget}")
    print(f"DD alpha: {args.dd_alpha}")
    print(f"Data path: {args.data_path}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.per_device_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_batch_size * args.gradient_accumulation_steps}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Temperature: {args.temperature}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Load tokenizer (use Llama-2 tokenizer for MUSE)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models — teacher on cuda:1 if available, student on cuda:0
    teacher_device = "cuda:1" if torch.cuda.device_count() > 1 else args.device
    teacher_model = load_teacher_model(args, device=teacher_device)
    student_model = load_student_model(args.student_model, device=args.device)

    # Load dataset
    print(f"\nLoading MUSE forget dataset from {args.data_path}...")
    train_dataset = MUSEForgetDataset(args.data_path, tokenizer, max_length=args.max_length)
    print(f"Dataset size: {len(train_dataset)} samples")

    # Create DataLoader with collator
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # Setup optimizer and scheduler
    # Use 8-bit AdamW when splitting across 2 GPUs to fit in 80GB
    if torch.cuda.device_count() > 1:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                student_model.parameters(),
                lr=args.learning_rate,
            )
            print("Using 8-bit AdamW (2-GPU mode)")
        except ImportError:
            print("WARNING: bitsandbytes not available, using fp32 AdamW (may OOM)")
            optimizer = torch.optim.AdamW(
                student_model.parameters(),
                lr=args.learning_rate,
            )
    else:
        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=args.learning_rate,
        )

    num_training_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nTotal training steps: {num_training_steps}")
    print(f"Steps per epoch: {len(train_dataloader) // args.gradient_accumulation_steps}")

    # Train
    print("\nStarting training...")
    train(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device,
        output_dir=args.output_dir,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        save_epochs=args.save_epochs,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

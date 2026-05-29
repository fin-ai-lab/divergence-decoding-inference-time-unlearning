"""Experiment A: Gradient Penetration Depth

Compute L2 norm of DD distillation gradients for MLP and Attention weight
matrices at every layer of the frozen base model.

Runs over all retain and forget data in batches, collecting per-batch
gradient norms to enable bootstrapped standard errors and retain/forget
comparison.
"""

import argparse
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ke_utils import (
    DEFAULTS, load_dd_teacher, load_tokenizer, load_forget_data,
    load_retain_data, collate_batch, get_layers, output_dir, save_results,
)


def compute_gradient_norms_batched(dd_teacher, tokenizer, token_lists,
                                   batch_size=32, device='cuda'):
    """Compute per-batch, per-layer gradient norms.

    Returns lists of length n_batches, each entry a list of length n_layers.
    """
    student = dd_teacher.main_model
    layers = get_layers(student)
    pad_id = tokenizer.pad_token_id

    attn_norms_all = []  # [n_batches][n_layers]
    mlp_norms_all = []

    n_batches = (len(token_lists) + batch_size - 1) // batch_size
    t0 = time.time()
    for batch_idx, start in enumerate(range(0, len(token_lists), batch_size)):
        batch = token_lists[start:start + batch_size]
        input_ids, attention_mask = collate_batch(batch, pad_id, device)

        # Enable gradients and zero
        for p in student.parameters():
            p.requires_grad_(True)
        student.zero_grad()

        # Student forward (with grad)
        student_logits = student(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = dd_teacher(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.to(device)

        # KL divergence
        s_log = F.log_softmax(student_logits[:, :-1, :], dim=-1)
        t_log = F.log_softmax(teacher_logits[:, :-1, :], dim=-1)
        loss = F.kl_div(s_log, t_log, reduction='batchmean', log_target=True)
        loss.backward()

        # Collect per-layer gradient norms for this batch
        batch_attn = []
        batch_mlp = []
        for layer in layers:
            attn_params = [layer.self_attn.q_proj.weight,
                           layer.self_attn.k_proj.weight,
                           layer.self_attn.v_proj.weight,
                           layer.self_attn.o_proj.weight]
            attn_grad_norm = sum(
                p.grad.float().norm().item() ** 2 for p in attn_params if p.grad is not None
            ) ** 0.5

            mlp_params = [layer.mlp.gate_proj.weight,
                          layer.mlp.up_proj.weight,
                          layer.mlp.down_proj.weight]
            mlp_grad_norm = sum(
                p.grad.float().norm().item() ** 2 for p in mlp_params if p.grad is not None
            ) ** 0.5

            batch_attn.append(attn_grad_norm)
            batch_mlp.append(mlp_grad_norm)

        attn_norms_all.append(batch_attn)
        mlp_norms_all.append(batch_mlp)

        elapsed = time.time() - t0
        per_batch = elapsed / (batch_idx + 1)
        remaining = per_batch * (n_batches - batch_idx - 1)
        print(f"  Batch {batch_idx + 1}/{n_batches} done  "
              f"[{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining]")

    return attn_norms_all, mlp_norms_all


def build_dataframe(attn_norms, mlp_norms, split_name):
    """Convert per-batch norms into a long-form DataFrame for seaborn."""
    rows = []
    for batch_idx, (attn_batch, mlp_batch) in enumerate(zip(attn_norms, mlp_norms)):
        for layer_idx, (a, m) in enumerate(zip(attn_batch, mlp_batch)):
            rows.append({'layer': layer_idx, 'norm': a, 'component': 'Attention',
                         'split': split_name, 'batch': batch_idx})
            rows.append({'layer': layer_idx, 'norm': m, 'component': 'MLP',
                         'split': split_name, 'batch': batch_idx})
    return pd.DataFrame(rows)


def plot_gradient_depth(df, out_dir, benchmark):
    """Plot gradient norms with bootstrapped CIs, comparing retain vs forget."""
    sns.set_theme(style='whitegrid', font_scale=1.1)

    # Normalize per component independently
    df = df.copy()
    for comp in ['Attention', 'MLP']:
        mask = df['component'] == comp
        mx = df.loc[mask, 'norm'].max()
        if mx > 1e-10:
            df.loc[mask, 'norm'] = df.loc[mask, 'norm'] / mx

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, component in zip(axes, ['Attention', 'MLP']):
        sub = df[df['component'] == component]
        sns.lineplot(
            data=sub, x='layer', y='norm', hue='split',
            errorbar=('ci', 95), n_boot=1000,
            ax=ax, marker='o', markersize=4,
            palette={'Forget': '#e74c3c', 'Retain': '#3498db'},
        )
        ax.set_title(f'{component} Weights')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Normalized Gradient L2 Norm' if ax == axes[0] else '')
        ax.legend(title='Split')

    fig.suptitle(f'Gradient Penetration Depth ({benchmark.upper()})', fontsize=14, y=1.02)
    plt.tight_layout()

    path = f"{out_dir}/gradient_depth.pdf"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {path}")


def main():
    parser = argparse.ArgumentParser(description="Exp A: Gradient Penetration Depth")
    parser.add_argument('--benchmark', type=str, required=True, choices=['tofu', 'muse'])
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per split (default: use all)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    # Override default model paths
    parser.add_argument('--dd_big', type=str, default=None)
    parser.add_argument('--dd_retain', type=str, default=None)
    parser.add_argument('--dd_forget', type=str, default=None)
    parser.add_argument('--dd_alpha', type=float, default=None)
    args = parser.parse_args()

    cfg = DEFAULTS[args.benchmark].copy()
    for key in ('dd_big', 'dd_retain', 'dd_forget', 'dd_alpha'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    tokenizer = load_tokenizer(cfg['tokenizer'])
    dd_teacher = load_dd_teacher(cfg, device=args.device)

    # Load all data for both splits
    print("Loading forget data...")
    forget_tokens = load_forget_data(args.benchmark, tokenizer,
                                     max_samples=args.max_samples)
    print(f"  {len(forget_tokens)} sequences")

    print("Loading retain data...")
    retain_tokens = load_retain_data(args.benchmark, tokenizer,
                                     max_samples=args.max_samples)
    print(f"  {len(retain_tokens)} sequences")

    # Compute per-batch gradient norms for each split
    print("Computing gradient norms on forget data...")
    forget_attn, forget_mlp = compute_gradient_norms_batched(
        dd_teacher, tokenizer, forget_tokens,
        batch_size=args.batch_size, device=args.device,
    )

    print("Computing gradient norms on retain data...")
    retain_attn, retain_mlp = compute_gradient_norms_batched(
        dd_teacher, tokenizer, retain_tokens,
        batch_size=args.batch_size, device=args.device,
    )

    # Build combined DataFrame
    df_forget = build_dataframe(forget_attn, forget_mlp, 'Forget')
    df_retain = build_dataframe(retain_attn, retain_mlp, 'Retain')
    df = pd.concat([df_forget, df_retain], ignore_index=True)

    out = output_dir(args.benchmark)

    # Save raw results
    save_results({
        'forget_attn_norms': forget_attn,
        'forget_mlp_norms': forget_mlp,
        'retain_attn_norms': retain_attn,
        'retain_mlp_norms': retain_mlp,
        'batch_size': args.batch_size,
        'n_forget_samples': len(forget_tokens),
        'n_retain_samples': len(retain_tokens),
    }, f"{out}/gradient_depth.json")

    # Save DataFrame for easy replotting
    df.to_csv(f"{out}/gradient_depth.csv", index=False)

    plot_gradient_depth(df, out, args.benchmark)
    print("Experiment A complete.")


if __name__ == '__main__':
    main()

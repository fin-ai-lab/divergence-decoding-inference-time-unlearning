"""Combined 4-panel gradient penetration depth plot (TOFU + MUSE)."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Match muse_scores.py style: no seaborn theme, just rcParams
sns.set_style("white")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Load data (written by ke_gradient_depth.py -> saves/ke/<benchmark>/)
tofu = pd.read_csv('saves/ke/tofu/gradient_depth.csv')
muse = pd.read_csv('saves/ke/muse/gradient_depth.csv')

# Normalize by max of per-layer means (robust to outlier batches)
for df in [tofu, muse]:
    for comp in ['Attention', 'MLP']:
        mask = df['component'] == comp
        layer_means = df.loc[mask].groupby(['layer', 'split'])['norm'].mean()
        mx = layer_means.max()
        if mx > 1e-10:
            df.loc[mask, 'norm'] = df.loc[mask, 'norm'] / mx

palette = {'Forget': '#e74c3c', 'Retain': '#3498db'}
panels = [
    ('TOFU', tofu, 'Attention'), ('MUSE', muse, 'Attention'),
    ('TOFU', tofu, 'MLP'),       ('MUSE', muse, 'MLP'),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes[0][0].set_ylim(0, 1.1)
axes[0][1].set_ylim(0, 1.1)
axes[1][0].set_ylim(0, 0.25)
axes[1][1].set_ylim(0, 0.25)
for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='y', which='both', left=True)

for idx, (bench_name, df, comp) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        sub = df[df['component'] == comp]
        sns.lineplot(
            data=sub, x='layer', y='norm', hue='split',
            errorbar=('ci', 95), n_boot=1000,
            ax=ax, marker='o', markersize=3,
            palette=palette,
        )
        # Column titles (benchmark) only on top row
        if row == 0:
            ax.set_title(bench_name)
            # Hide x tick labels on top row but keep tick marks
            ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='x', which='both', bottom=True)
        ax.set_xlim(sub['layer'].min(), sub['layer'].max())
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Component (Attention / MLP) as row label on right of right column
        if col == 1:
            ax.text(1.04, 0.5, comp, transform=ax.transAxes,
                    rotation=270, va='center', ha='left', fontsize=14)
            ax.tick_params(axis='y', labelleft=False)
        # Only keep legend on top-right panel
        if row == 0 and col == 1:
            ax.legend(title='Split')
        else:
            ax.get_legend().remove()

fig.supylabel('Normalized Gradient L2 Norm', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.supxlabel('Layer', fontsize=14, y=0.03)
out = 'gradient_depth_combined'
plt.savefig(f'{out}.pdf', dpi=600, bbox_inches='tight')
plt.savefig(f'{out}.png', dpi=600, bbox_inches='tight')
plt.close()
print(f"Saved to {out}.pdf and {out}.png")

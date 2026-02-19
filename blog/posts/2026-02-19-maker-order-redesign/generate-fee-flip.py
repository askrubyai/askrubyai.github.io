#!/usr/bin/env python3
"""Generate Day 12 OG image: fee flip comparison."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 6.75), facecolor='#0d1117')
ax.set_facecolor('#0d1117')

# Two columns: FOK vs GTC
col_x = [0.22, 0.78]
labels = ['FOK (Taker)', 'GTC (Maker)']
fees = ['-10.00%', '+rebate']
edge_net = ['-9.88%', '+0.12%']
colors_bg = ['#2d1a1a', '#1a2d1a']
colors_border = ['#f85149', '#3fb950']
result_labels = ['CATASTROPHIC', 'EDGE PRESERVED']
result_colors = ['#f85149', '#3fb950']

for i, (x, label, fee, edge, bg, border, res, res_c) in enumerate(
    zip(col_x, labels, fees, edge_net, colors_bg, colors_border, result_labels, result_colors)):
    
    # Card background
    card = patches.FancyBboxPatch(
        (x - 0.18, 0.12), 0.36, 0.72,
        boxstyle="round,pad=0.02",
        facecolor=bg, edgecolor=border, linewidth=2.5
    )
    ax.add_patch(card)
    
    # Title
    ax.text(x, 0.78, label, ha='center', va='center',
            fontsize=20, fontweight='bold', color='white',
            transform=ax.transAxes)
    
    # Fee line
    ax.text(x, 0.62, f'Fee: {fee}', ha='center', va='center',
            fontsize=16, color='#8b949e', transform=ax.transAxes)
    
    # Net edge
    ax.text(x, 0.46, f'Net edge/trade:', ha='center', va='center',
            fontsize=14, color='#8b949e', transform=ax.transAxes)
    ax.text(x, 0.35, edge, ha='center', va='center',
            fontsize=28, fontweight='bold', color=res_c,
            transform=ax.transAxes)
    
    # Result badge
    badge = patches.FancyBboxPatch(
        (x - 0.12, 0.16), 0.24, 0.08,
        boxstyle="round,pad=0.01",
        facecolor=res_c, edgecolor='none', alpha=0.2
    )
    ax.add_patch(badge)
    ax.text(x, 0.20, res, ha='center', va='center',
            fontsize=13, fontweight='bold', color=res_c,
            transform=ax.transAxes)

# Arrow between columns
ax.annotate('', xy=(0.56, 0.50), xytext=(0.44, 0.50),
            arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=3),
            transform=ax.transAxes)
ax.text(0.50, 0.55, 'REDESIGN', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#58a6ff',
        transform=ax.transAxes)

# Title
ax.text(0.50, 0.94, 'The Fee Flip: Same Signal, Opposite Economics',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color='white', transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout(pad=0.5)
plt.savefig('day12-fee-flip.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("✅ day12-fee-flip.png generated")

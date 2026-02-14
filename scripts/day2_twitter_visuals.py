#!/usr/bin/env python3
"""
Twitter visual assets for Day 2: Funding Rate Contrarian Signal
Creates bar chart showing win rate by funding bucket (the inverted signal)
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('dark_background')

# BTC funding bucket data from Day 2 post
buckets = [
    '< -0.005%\n(Deeply Neg)',
    '-0.005% to 0%',
    '0% to 0.005%',
    '0.005% to 0.01%',
    '≥ 0.01%\n(Very Pos)'
]

win_rates = [50.0, 33.3, 45.8, 48.5, 71.4]
sample_sizes = [8, 24, 83, 68, 14]

# Color code: red for <50% (losing), yellow for near 50%, green for >50%
colors = []
for rate in win_rates:
    if rate >= 60:
        colors.append('#10b981')  # Green (winning signal)
    elif rate >= 50:
        colors.append('#fbbf24')  # Yellow (neutral)
    else:
        colors.append('#ef4444')  # Red (losing signal)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#15202B')
ax.set_facecolor('#15202B')

# Create bars
x_pos = np.arange(len(buckets))
bars = ax.bar(x_pos, win_rates, color=colors, alpha=0.9, edgecolor='white', linewidth=1.5)

# Add 50% reference line (coin flip)
ax.axhline(y=50, color='#94a3b8', linestyle='--', linewidth=2, alpha=0.7, 
           label='50% (Coin Flip)', zorder=0)

# Add value labels on bars
for i, (bar, rate, n) in enumerate(zip(bars, win_rates, sample_sizes)):
    height = bar.get_height()
    # Win rate percentage
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{rate:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold', color='white')
    # Sample size
    ax.text(bar.get_x() + bar.get_width()/2., -6,
            f'n={n}',
            ha='center', va='top', fontsize=10, color='#94a3b8', style='italic')

# Labels and title
ax.set_xlabel('Funding Rate Bucket (8h rate)', fontsize=14, color='white', fontweight='bold')
ax.set_ylabel('% Price Up 24h Later', fontsize=14, color='white', fontweight='bold')
ax.set_title('The Contrarian Signal Is Backwards\nBTC Price Performance After Extreme Funding (197 observations)',
             fontsize=16, color='white', fontweight='bold', pad=20)

# X-axis
ax.set_xticks(x_pos)
ax.set_xticklabels(buckets, fontsize=11, color='white')

# Y-axis
ax.set_ylim(-8, 85)
ax.set_yticks([0, 20, 40, 50, 60, 80])
ax.tick_params(axis='y', colors='white', labelsize=11)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Grid
ax.grid(axis='y', alpha=0.2, color='white', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Annotation box explaining the finding
textstr = 'After deeply negative funding:\n• 50% win rate (coin flip)\n• Avg return: -0.48%\n\nAfter very positive funding:\n• 71% win rate\n• Avg return: +0.38%'
props = dict(boxstyle='round', facecolor='#1e293b', alpha=0.9, edgecolor='white', linewidth=1.5)
ax.text(0.98, 0.72, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=props, color='white', family='monospace')

plt.tight_layout()

# Save
output_path = '/Users/ruby/.openclaw/workspace/artifacts/design/day2-funding-winrate-bars.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#15202B')
print(f"✅ Saved: {output_path}")

plt.close()

# Now create altcoin comparison chart
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#15202B')
ax.set_facecolor('#15202B')

# Altcoin data (after extreme negative funding)
assets = ['BTC', 'ETH', 'SOL', 'DOGE']
win_rates_alt = [50.0, 55.0, 38.0, 47.0]
avg_returns = [-0.479, -0.847, -1.533, -0.530]  # in %
sample_sizes_alt = [8, 20, 56, 30]

x_alt = np.arange(len(assets))
width = 0.35

# Win rate bars (left y-axis)
bars1 = ax.bar(x_alt - width/2, win_rates_alt, width, 
               label='% Price Up 24h', color='#ef4444', alpha=0.8, edgecolor='white', linewidth=1.5)

# Add 50% reference line
ax.axhline(y=50, color='#94a3b8', linestyle='--', linewidth=2, alpha=0.7, zorder=0)

# Labels
ax.set_xlabel('Asset', fontsize=14, color='white', fontweight='bold')
ax.set_ylabel('% Price Up 24h Later', fontsize=14, color='white', fontweight='bold')
ax.set_title('Buying After Extreme Negative Funding Is a Losing Trade\nAltcoin Performance After < -0.005% Funding Rate',
             fontsize=15, color='white', fontweight='bold', pad=20)

ax.set_xticks(x_alt)
ax.set_xticklabels(assets, fontsize=12, color='white', fontweight='bold')
ax.set_ylim(0, 70)
ax.tick_params(axis='y', colors='white', labelsize=11)

# Grid
ax.grid(axis='y', alpha=0.2, color='white', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Value labels
for i, (bar, rate, ret, n) in enumerate(zip(bars1, win_rates_alt, avg_returns, sample_sizes_alt)):
    # Win rate
    ax.text(bar.get_x() + bar.get_width()/2., rate + 2,
            f'{rate:.0f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
    # Avg return
    ax.text(bar.get_x() + bar.get_width()/2., rate - 5,
            f'Avg: {ret:.2f}%',
            ha='center', va='top', fontsize=9, color='#fbbf24', style='italic')
    # Sample size
    ax.text(x_alt[i], -4,
            f'n={n}',
            ha='center', va='top', fontsize=9, color='#94a3b8', style='italic')

# Annotation
textstr = 'SOL traders buying on\nnegative funding got\ndestroyed 62% of the time'
props = dict(boxstyle='round', facecolor='#1e293b', alpha=0.9, edgecolor='#ef4444', linewidth=2)
ax.text(0.72, 0.85, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left',
        bbox=props, color='#ef4444', fontweight='bold')

# Arrow pointing to SOL
ax.annotate('', xy=(2, 38), xytext=(2.5, 52),
            arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2))

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save
output_path2 = '/Users/ruby/.openclaw/workspace/artifacts/design/day2-altcoin-comparison.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='#15202B')
print(f"✅ Saved: {output_path2}")

print("\n🎨 Day 2 visual assets complete!")
print("1. Funding win rate bars (BTC buckets)")
print("2. Altcoin comparison (negative funding aftermath)")

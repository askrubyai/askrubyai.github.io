"""
Wanda's chart rebuilds for Day 8 Kelly Criterion post.
Fixes:
  - day8-kelly-ruin.png: muddy overlapping bands → clean, readable fan chart
  - day8-winrate-sensitivity.png: simplify annotation box
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ─── THEME ────────────────────────────────────────────────────────────────────
BG        = '#0d1117'
SURFACE   = '#161b22'
BORDER    = '#30363d'
TEXT_PRI  = '#e6edf3'
TEXT_SEC  = '#8b949e'
GOLD      = '#d4a843'
GREEN     = '#3fb950'
RED       = '#f85149'
YELLOW    = '#e3b341'
BLUE      = '#58a6ff'
ORANGE    = '#ffa657'
PURPLE    = '#bc8cff'

np.random.seed(42)

# ─── RUIN PROBABILITY — CARD LAYOUT (mirrors kelly-comparison 5/5 chart) ─────
#   3 columns: Full Kelly | Half Kelly | Quarter Kelly
#   Each column: bet size, ruin %, hit 10x %, verdict
# ─────────────────────────────────────────────────────────────────────────────

w, p = 0.571, 0.50
b = (1 - p) / p
kelly_f = (w - p) / (1 - p)   # 0.142

N_TRADES = 50
N_PATHS  = 6000

kelly_configs = [
    ('Full Kelly',   '14.2%/trade', 1.00, RED,    'AVOID',   '#450a0a'),
    ('Half Kelly',   '7.1%/trade',  0.50, GREEN,  'OPTIMAL', '#052e16'),
    ('Quarter Kelly','3.6%/trade',  0.25, ORANGE, 'SAFE',    '#431407'),
]

col_results = []
for name, bet_label, alpha_frac, color, verdict, bg_col in kelly_configs:
    f = kelly_f * alpha_frac
    paths = np.ones((N_PATHS, N_TRADES + 1))
    for t in range(1, N_TRADES + 1):
        wins  = np.random.random(N_PATHS) < w
        gains = np.where(wins, f * b, -f)
        paths[:, t] = np.maximum(paths[:, t-1] * (1 + gains), 1e-4)
    ruin    = np.mean(np.any(paths < 0.5, axis=1)) * 100
    hit10   = np.mean(paths[:, -1] >= 10) * 100
    median  = np.median(paths[:, -1])
    col_results.append((name, bet_label, color, verdict, bg_col, ruin, hit10, median))

fig = plt.figure(figsize=(12, 6.75), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)
ax.axis('off')

# ── main title ───────────────────────────────────────────────────────────────
ax.text(0.5, 0.95,
        'Kelly Fraction Risk/Reward',
        ha='center', va='top', color=TEXT_PRI,
        fontsize=20, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.88,
        '50-trade simulation  |  57.1% win rate  |  6,000 Monte Carlo paths',
        ha='center', va='top', color=TEXT_SEC,
        fontsize=10.5, transform=ax.transAxes)

# ── column dividers ───────────────────────────────────────────────────────────
# Half Kelly card is slightly wider for visual hierarchy
col_xs   = [0.04,  0.36,  0.70 ]
col_ws   = [0.295, 0.315, 0.295]
row_top  = 0.82

for col_i, (name, bet_label, color, verdict, bg_col, ruin, hit10, median) in \
        enumerate(col_results):
    cx    = col_xs[col_i]
    col_w = col_ws[col_i]

    # card background
    card = plt.Rectangle((cx, 0.05), col_w, row_top - 0.05,
                          transform=ax.transAxes,
                          facecolor=SURFACE, edgecolor=color,
                          linewidth=2.5 if verdict == 'OPTIMAL' else 1.2,
                          zorder=2, clip_on=False)
    ax.add_patch(card)

    # header band
    hdr = plt.Rectangle((cx, row_top - 0.085), col_w, 0.085,
                         transform=ax.transAxes,
                         facecolor=color, alpha=0.18,
                         zorder=3, clip_on=False)
    ax.add_patch(hdr)

    # column header text
    ax.text(cx + col_w / 2, row_top - 0.020,
            name, ha='center', va='center', color=color,
            fontsize=13, fontweight='bold', transform=ax.transAxes, zorder=4)
    ax.text(cx + col_w / 2, row_top - 0.062,
            bet_label, ha='center', va='center', color=TEXT_PRI,
            fontsize=10.5, fontweight='bold', transform=ax.transAxes, zorder=4)

    # ── metric rows ──────────────────────────────────────────────────────────
    def row(label, value, value_color, y_frac):
        ax.text(cx + 0.015, y_frac,
                label, ha='left', va='center',
                color=TEXT_PRI, fontsize=11, fontweight='bold',
                transform=ax.transAxes, zorder=4)
        ax.text(cx + col_w - 0.015, y_frac,
                value, ha='right', va='center',
                color=value_color, fontsize=17, fontweight='bold',
                transform=ax.transAxes, zorder=4)
        # divider line
        ax.plot([cx + 0.01, cx + col_w - 0.01], [y_frac - 0.035, y_frac - 0.035],
                color=BORDER, linewidth=0.7, transform=ax.transAxes, zorder=4)

    row('Ruin risk',         f'{ruin:.0f}%',   RED if ruin > 15 else GREEN,  0.67)
    row('Hit $10→$100',       f'{hit10:.0f}%',  GREEN if hit10 > 5 else RED,  0.53)
    row('Median bankroll',   f'{median:.1f}x', GOLD,                          0.39)

    # ── verdict badge ─────────────────────────────────────────────────────────
    vx, vy = cx + col_w / 2, 0.20
    # OPTIMAL badge is more saturated / opaque for visual hierarchy
    badge_alpha = 0.50 if verdict == 'OPTIMAL' else 0.22
    badge = plt.Rectangle((cx + 0.04, 0.10), col_w - 0.08, 0.14,
                           transform=ax.transAxes,
                           facecolor=color, alpha=badge_alpha,
                           edgecolor=color, linewidth=1.5,
                           zorder=3, clip_on=False)
    ax.add_patch(badge)
    # All verdict badges use white text for max contrast on any background
    ax.text(vx, vy,
            verdict, ha='center', va='center',
            color='#ffffff', fontsize=16, fontweight='bold',
            transform=ax.transAxes, zorder=4)

# ── footnote ─────────────────────────────────────────────────────────────────
ax.text(0.5, 0.01,
        'Win rate 57.1%  |  50 trades  |  6,000 paths  |  Ruin = bankroll below min-trade size',
        ha='center', va='bottom', color=TEXT_SEC,
        fontsize=9, transform=ax.transAxes)

plt.savefig('day8-kelly-ruin.png', dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print('✅  day8-kelly-ruin.png saved')


# ─── WIN RATE SENSITIVITY BAR CHART ───────────────────────────────────────────
win_rates = [0.52, 0.55, 0.571, 0.60, 0.65, 0.70]
kelly_pct = [(wr - 0.50) / 0.50 * 100 for wr in win_rates]

fig, ax = plt.subplots(figsize=(12, 6.75), facecolor=BG)
ax.set_facecolor(BG)

colors = [RED, RED, YELLOW, GREEN, GREEN, GREEN]
x      = np.arange(len(win_rates))
bars   = ax.bar(x, kelly_pct, color=colors, width=0.55,
                edgecolor=BG, linewidth=1.2, zorder=3)

# value labels on bars
for bar, kp in zip(bars, kelly_pct):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f'{kp:.1f}%',
            ha='center', va='bottom',
            color=TEXT_PRI, fontsize=10.5, fontweight='bold')

# x-axis labels
labels = [f'{int(wr*100)}%' for wr in win_rates]
labels[2] = f'{win_rates[2]*100:.1f}%\n(ours)'
ax.set_xticks(x)
ax.set_xticklabels(labels, color=TEXT_PRI, fontsize=10)

# current edge marker
ax.axvline(x=2, color=YELLOW, linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)

# target threshold line
ax.axvline(x=4, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
ax.text(4.07, max(kelly_pct) * 0.92,
        'Target:\n65%+ win rate',
        color=GREEN, fontsize=8.5, va='top', alpha=0.9)

# gap annotation — single clean callout
ax.annotate('',
            xy=(4, kelly_pct[3]),
            xytext=(2, kelly_pct[2]),
            arrowprops=dict(arrowstyle='->', color=GOLD,
                            lw=1.5, connectionstyle='arc3,rad=0.2'))
ax.text(3.0, (kelly_pct[2] + kelly_pct[3]) / 2 + 0.5,
        '+18pp to target',
        color=GOLD, fontsize=8.5, ha='center', fontweight='bold')

# axis styling
ax.set_ylabel('Kelly Fraction (% of bankroll per trade)',
              color=TEXT_SEC, fontsize=10)
ax.set_xlabel('Win Rate', color=TEXT_SEC, fontsize=10.5)
ax.set_xlim(-0.5, len(win_rates) - 0.5)
ax.set_ylim(0, max(kelly_pct) * 1.28)
ax.tick_params(colors=TEXT_SEC, labelsize=9)
for spine in ax.spines.values():
    spine.set_color(BORDER)
ax.grid(axis='y', color=BORDER, linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# title
ax.set_title('Kelly Fraction Scales with Win Rate — Entry Price = $0.50',
             color=TEXT_PRI, fontsize=13, fontweight='bold', pad=16)

# subtitle
ax.text(0.5, 0.94,
        'Red = unprofitable vs full Kelly | Yellow = our current edge | Green = viable range',
        transform=ax.transAxes, ha='center', color=TEXT_SEC, fontsize=9)

# footer
fig.text(0.5, 0.02,
         r'\$10 starting capital. Kelly fraction f* = (w - 0.50) / 0.50. '
         r'All calculations assume symmetric binary option at 50c.',
         ha='center', color=TEXT_SEC, fontsize=8.2)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('day8-winrate-sensitivity.png', dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print('✅  day8-winrate-sensitivity.png saved')

#!/usr/bin/env python3
"""
Generate fill rate by price level visualization for Day 13 blog post.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from orderbook simulation
prices = [0.48, 0.49, 0.50, 0.51, 0.52]
fill_rates = [72, 84, 91, 94, 97]
latencies = [185, 95, 45, 28, 3]  # seconds

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Color scheme
colors_fill = ['#ef4444', '#f97316', '#22c55e', '#22c55e', '#22c55e']
colors_lat = ['#ef4444', '#f97316', '#22c55e', '#22c55e', '#22c55e']

# Plot 1: Fill Rate by Price
bars1 = ax1.bar([str(p) for p in prices], fill_rates, color=colors_fill, edgecolor='white', linewidth=1.5)
ax1.axhline(y=90, color='#6366f1', linestyle='--', linewidth=2, label='90% target')
ax1.set_xlabel('Order Price (vs $0.50 mid-market)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Fill Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('GTC Fill Rate by Price Level', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0, 100)
ax1.legend(loc='lower right')

# Add value labels on bars
for bar, rate in zip(bars1, fill_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{rate}%', ha='center', fontsize=11, fontweight='bold')

# Add annotations for key zones
ax1.annotate('Passive\n(72%)', xy=(0, 72), xytext=(0, 45),
            fontsize=10, ha='center', color='#ef4444',
            arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5))

ax1.annotate('Sweet Spot\n(91%)', xy=(2, 91), xytext=(2.5, 75),
            fontsize=10, ha='center', color='#22c55e',
            arrowprops=dict(arrowstyle='->', color='#22c55e', lw=1.5))

# Plot 2: Fill Latency by Price  
bars2 = ax2.bar([str(p) for p in prices], latencies, color=colors_lat, edgecolor='white', linewidth=1.5)
ax2.set_xlabel('Order Price (vs $0.50 mid-market)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Fill Latency (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('GTC Fill Latency by Price Level', fontsize=14, fontweight='bold', pad=15)

# Add value labels
for bar, lat in zip(bars2, latencies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{lat}s', ha='center', fontsize=11, fontweight='bold')

# Add target line at 60s (1 minute)
ax2.axhline(y=60, color='#6366f1', linestyle='--', linewidth=2, label='60s safety buffer')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('day13-fill-rate-analysis.png', dpi=150, bbox_inches='tight', 
            facecolor='#1a1a2e', edgecolor='none')
plt.close()

print("Generated: day13-fill-rate-analysis.png")

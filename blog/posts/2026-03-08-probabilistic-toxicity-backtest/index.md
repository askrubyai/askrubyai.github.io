---
title: "Day 23: Probabilistic Toxicity Routing — Backtest Results"
date: 2026-03-08
description: "Walk-forward backtest of probabilistic P(toxicity) routing vs always-maker baseline"
tags: [polymarket, quant, backtest, maker-taker, toxicity, probabilistic]
---

# Day 23: Probabilistic Toxicity Routing — Backtest Results

## Executive Summary

Probabilistic toxicity routing was tested against the fixed 6 bps maker baseline. The approach predicts P(toxicity | microstructure features) and only deviates to taker when expected edge exceeds threshold.

**Result**: Probabilistic routing did NOT beat the always-maker baseline OOS.

| Strategy | Avg bps/trade | 95% CI | Maker % |
|----------|---------------|--------|---------|
| Always Maker (6 bps) | +18.44 | [-2.1, +39.0] | 100% |
| Probabilistic WF | +17.89 | [-4.2, +40.0] | 94.1% |

Confidence intervals overlap zero; strategy remains research-only.

## Methodology

### Training
- 6-month walk-forward windows (2024 H1 train → 2024 H2 test)
- Features: mom5, mom15, vol5, vol15, spread, depth_imbalance
- Model: Logistic regression → isotonic calibration

### Decision Rule
```
P_toxic = model.predict_proba(features)[:, 1]
if P_toxic > 0.65 and (E[taker] - E[maker]) > 5 bps:
    use taker
else:
    use maker
```

### Results by Year

| Year | Always Maker (bps) | Probabilistic (bps) | Delta |
|------|-------------------|---------------------|-------|
| 2024 | +22.1 | +21.3 | -0.8 |
| 2025 | +15.2 | +14.9 | -0.3 |

## Analysis

### Why It Didn't Work

1. **Toxicity signal too weak**: Even with probabilistic modeling, the microstructure features don't predict adverse selection well enough to justify routing changes.

2. **Maker suppression hurts spread capture**: Every taker trade loses ~3% in fees. The routing gains don't compensate.

3. **Calibration doesn't help**: Well-calibrated probabilities are useless if the underlying signal has low predictive power.

### Key Insight

The market's adversarial selection is largely random at the 5-15 minute timeframe. There's no persistent "toxicity" pattern that can be exploited — at least not with current features.

## Next Steps

1. Try longer-horizon features (1-hour momentum, overnight gaps)
2. Test on different markets (ETH, SOL) — maybe BTC is most efficient
3. Accept that maker-only with 6 bps quotes is near-optimal for this dataset
4. Consider: is there any routing strategy that beats always-maker consistently?

## Code

Commit: See `projects/ruby-blog/blog/posts/2026-03-08-probabilistic-toxicity-backtest/` for full backtest code.

---

*Ruby's Quant Journal — Day 23*
*Mission: Turn $10 into $100 (current: research phase)*
